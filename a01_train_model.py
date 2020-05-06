#!/usr/bin/env python3
from prepare_model import train_loop


def main():
    train_loop(
        batch_size=64,
        test_batch_size=1000,
        epochs=14,
        lr=1.0,
        gamma=0.7,
        no_cuda=True,
        seed=1,
        log_interval=10,
        save_model=True
    )


if __name__ == '__main__':
    main()
