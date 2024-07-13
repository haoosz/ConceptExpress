python infer.py \
  --embed_path ckpts/69/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the beach" \
  --save_path output/69 \
  --seed 0

python infer.py \
  --embed_path ckpts/69/learned_embeds_final.bin \
  --prompt "a photo of <asset1> in the snow" \
  --save_path output/69 \
  --seed 0