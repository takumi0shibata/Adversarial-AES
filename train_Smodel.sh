#!/usr/bin/env bash
for mode in 'domain-adaptation' 'source'
do
    for prompt in {1..8}
    do
        python train_Smodel.py --test_prompt_id ${prompt} --train_mode ${mode}
    done
done