# Wasserstein Domain Generalization
Details coming soon


An example for experiments on PACS choosing Sketch as target:

   python3 main.py --target S --lr_fea 2e-4 --lr_clf 2e-4 --lr_dis 2e-4 \
         --lr_mtr 1e-5 --weight_cls_loss 1 --weight_dis_loss 1 --weight_decay 1e-5 \
          --wd_round 1 --weight_mtr_loss 1e-6 --mtr_margin 2 \
          --mtr_scale_pos 2 --mtr_scale_neg 40 --gp_param 0.001 --add_clsuter 5
        
