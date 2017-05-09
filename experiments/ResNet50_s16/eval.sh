python ../../tools/test_net.py --gpu 15 \
--def tt100k_test_agnostic_s16_rpn15.prototxt \
--net ../../output/rfcn_end2end_ohem/tt100ktrain/resnet50_rfcn_ohem_iter_40000.caffemodel \
--cfg ./config.yml \
--imdb tt100k_test
