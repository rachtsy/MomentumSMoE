import warnings

warnings.filterwarnings("ignore")

import math
import torch
import time

from config import PARAMS_CONFIG
from data import get_train_val_test_data
from models import TransformerSeq
from trainer import train_iteration, full_eval
import datetime
import wandb
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    freeze_gate_weight,
    Logger,
    set_freq_optimal_search,
)


def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    wandb_flag = wandb_params["wandb_flag"]
    if wandb_flag:
        wandb.init(project=wandb_params["project_name"])
        wandb.run.name = wandb_params["job_name"]
        wandb.config.update(model_params)
    # global val
    best_val_loss = None
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]
    resume = trainer_params["resume"]

    if distributed == False or env_params["rank"] == 0:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)

    # DATA
    train_data, val_data, test_data = get_train_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params["batch_size"],
        device=device,
    )

    # MODEL
    model = TransformerSeq(
        vocab_size=data_params["vocab_size"],
        **model_params,
        adapt_span_params=adapt_span_params,
    )
    print(model)
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # create logger
    logger = Logger()
    fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
    logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")
    # log paramters
    logging(f"Training Parameters:\n {trainer_params}")
    logging(f"Models Parameters:\n {model_params}")
    # logging time
    current_time = datetime.datetime.now()
    logging(str(current_time))
    # log model
    logging(str(model))
    logging(f"Total of Parameters: {sum(p.numel() for p in model.parameters())}")
    logging(
        f"Total of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume,
    )
    # fix gate
    if model_params["smoe_dropout"]:
        freeze_gate_weight(model)
    # calculate time
    start_time = time.time()
    # eval model
    if trainer_params["full_eval_mode"]:
        # evaluate the model on test data
        with torch.no_grad():
            loss_val = full_eval(
                model,
                optimizer,
                scheduler,
                val_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            loss_test = full_eval(
                model,
                optimizer,
                scheduler,
                test_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                torch.distributed.reduce(stats, 0)
                if env_params["rank"] == 0:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                else:
                    return

            # print('Test BPC: {:.4f}'.format(loss_test / math.log(2)))
            if ("enwik8" in data_params["data_path"]) or (
                "text8" in data_params["data_path"]
            ):
                logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
                logging("Test: {:.3f} BPC".format(loss_test / math.log(2)))
            else:
                logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
                logging("Test: {:.3f} PPL".format(math.exp(loss_test)))
        return

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [
        [
            torch.zeros(
                train_data.size(0),
                model.module.layers[layer_i].attn.attn.get_cache_size(),
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(model.module.attn_layer_count)
        ]
        for _ in range(2)
    ]

    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    for iter_no in range(iter_init, trainer_params["nb_iter"]):
        # freq type
        if model_params["freq_type"] == "function":
            _threshold = 2.0 / (2.0 + math.sqrt((iter_no + 1)))
            set_freq_optimal_search(model, _threshold)

        # time storing
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0] = train_iteration(
            model,
            model_params["load_balance"],
            optimizer,
            scheduler,
            train_data,
            nb_batches_per_iter,
            model_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            trainer_params["checkpoint_path"],
        )
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1] = train_iteration(
                model,
                model_params["load_balance"],
                optimizer,
                scheduler,
                val_data,
                nb_batches_per_iter,
                model_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                trainer_params["checkpoint_path"],
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue
        logging(f"=================== EPOCHS {iter_no} ======================")
        if ("enwik8" in data_params["data_path"]) or (
            "text8" in data_params["data_path"]
        ):
            msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} BPC | loss_val: {:.3f} ~ {:.3f} BPC | elapsed: {:.1f}".format(
                iter_no,
                loss_train,
                float(loss_train / math.log(2)),
                loss_val,
                float(loss_val / math.log(2)),
                elapsed,
            )
        else:
            msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} PPL | loss_val: {:.3f} ~ {:.3f} PPL | elapsed: {:.1f}".format(
                iter_no,
                loss_train,
                float(math.exp(loss_train)),
                loss_val,
                float(math.exp(loss_val)),
                elapsed,
            )
        logging(msg_result)
        if wandb_flag:
            wandb.log({'train_ppl':float(math.exp(loss_train)),'Epoch':iter_no,'valid_ppl':float(math.exp(loss_val))})
        logger.log_iter(iter_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model)
        # Save the model if the validation loss is the best we've seen so far.
        if (best_val_loss is None) or loss_val < best_val_loss:
            best_val_loss = loss_val
            save_checkpoint(
                trainer_params["checkpoint_path"],
                iter_no,
                model,
                optimizer,
                scheduler,
                logger,
            )
        # save_checkpoint(trainer_params['checkpoint_path'], nb_batches_per_iter, model, optimizer, scheduler, logger)
    end_time = time.time()
    logging(f"Training time total: {(end_time - start_time)/3600} h")


if __name__ == "__main__":
    launch(**get_params(params_config=PARAMS_CONFIG))
