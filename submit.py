def main():
    from pathlib import Path

    import submitit

    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        slurm_partition="gpu,a100,r7525,xe8545,tmp",
        slurm_gpus_per_task=1,
        slurm_additional_parameters={"gpu-bind": "single:1"},
        name="jax_privacy",
        slurm_time="02:00:00",
    )
    fn = submitit.helpers.CommandFunction(
        [
            "uv",
            "run",
            "python",
            "-m",
            "experiments.image_classification.run_experiment_loop",
            "--config=experiments/image_classification/configs/cifar10_wrn_16_4_eps1.py",
        ],
        cwd=Path(__file__).parent,
    )
    job = executor.submit(fn)
    print(job.result())


if __name__ == "__main__":
    main()
