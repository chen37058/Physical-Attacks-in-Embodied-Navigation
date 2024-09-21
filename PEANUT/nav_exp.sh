nohup python -u nav/eval.py \
            -v 2 \
            --num_sem_categories 10 \
            --pred_model_cfg "nav/pred_model_cfg.py" \
            --pred_model_wts "test_patch/models/pred_model_wts.pth" \
            --seg_model_wts "test_patch/models/mask_rcnn_R_101_cat9.pth" \
            --evaluation 'local' \
            --dump_location test_patch/result \
            --exp_name debug \
            &>log.log