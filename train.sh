
echo "============================================"
echo "anchor_based, canonical setting, tvsum&summe:"
python train.py --model anchor-based --model-dir ./models/ab_basic --splits ./splits/tvsum.yml ./splits/summe.yml
