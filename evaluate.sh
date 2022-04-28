
echo "============================================"
echo "anchor_based, canonical setting, tvsum&summe:"
python evaluate.py --model anchor-based --model-dir ./eval_models/anchor_based/canonical --splits ./splits/tvsum.yml ./splits/summe.yml
echo "============================================"
echo "anchor_based, augmented setting, tvsum&summe:"
python evaluate.py --model anchor-based --model-dir ./eval_models/anchor_based/augmented --splits ./splits/tvsum_aug.yml ./splits/summe_aug.yml
echo "============================================"
echo "anchor_based, transfer setting, tvsum&summe:"
python evaluate.py --model anchor-based --model-dir ./eval_models/anchor_based/transfer --splits ./splits/tvsum_trans.yml ./splits/summe_trans.yml
echo "============================================"
echo "anchor_free, canonical setting, tvsum&summe:"
python evaluate.py --model anchor-free --model-dir ./eval_models/anchor_free/canonical --splits ./splits/tvsum.yml ./splits/summe.yml