from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor

from configs import get_cfg
from dataloader import dataset
from engine.trainer import get_trainer_class
import custom_meta_arch  # need this to register meta-arch


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    TrainClass = get_trainer_class(cfg)
    default_setup(cfg, args)  # mkdir, config, logger
    dataset.register_dataset(cfg)

    if args.eval_only:
        predictor = DefaultPredictor(cfg)
        res = TrainClass.test(cfg, predictor.model)
        return res

    trainer = TrainClass(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
