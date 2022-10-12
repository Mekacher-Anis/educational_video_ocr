"""
This has been copied from $MMOCR_HOME/mmocr/ocr.py
and has been modified to include the checkpoints of the
finetuned models on the lvdb dataset.
"""

model_dict = {
            'Tesseract': {},
            # Detection models
            'DB_r18': {
                'config': 'textdet/dbnet/dbnet_resnet18_fpnc_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/dbnet_resnet18_fpnc_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'DB_r50': {
                'config': 'textdet/dbnet/dbnet_resnet50-dcnv2_fpnc_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/dbnet_resnet50-dcnv2_fpnc_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'DBPP_r50': {
                'config': 'textdet/dbnetpp/dbnetpp_resnet50-dcnv2_fpnc_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/dbnetpp_resnet50-dcnv2_fpnc_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'DRRG': {
                'config':
                'textdet/'
                'drrg/drrg_resnet50_fpn-unet_20e_lvdb.py',
                'ckpt':
                'textdet/'
                'drrg/'
                'drrg_resnet50_fpn-unet_1200e_ctw1500/'
                'drrg_resnet50_fpn-unet_1200e_ctw1500_20220827_105233-d5c702dd.pth'  # noqa: E501
            },
            'FCE_IC15': {
                'config':
                'textdet/'
                'fcenet/fcenet_resnet50_fpn_20e_lvdb.py',
                'ckpt':
                'textdet/'
                'fcenet/'
                'fcenet_resnet50_fpn_1500e_icdar2015/'
                'fcenet_resnet50_fpn_1500e_icdar2015_20220826_140941-167d9042.pth'  # noqa: E501
            },
            'FCE_CTW_DCNv2': {
                'config': 'textdet/fcenet/fcenet_resnet50-dcnv2_fpn_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/fcenet_resnet50-dcnv2_fpn_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'MaskRCNN_CTW': {
                'config':
                'textdet/'
                'maskrcnn/mask-rcnn_resnet50_fpn_160e_ctw1500.py',
                'ckpt':
                'textdet/'
                'maskrcnn/'
                'mask-rcnn_resnet50_fpn_160e_ctw1500/'
                'mask-rcnn_resnet50_fpn_160e_ctw1500_20220826_154755-ce68ee8e.pth'  # noqa: E501
            },
            'MaskRCNN_IC15': {
                'config': 'textdet/maskrcnn/mask-rcnn_resnet50_fpn_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/mask-rcnn_resnet50_fpn_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'PANet_CTW': {
                'config':
                'textdet/'
                'panet/panet_resnet18_fpem-ffm_600e_ctw1500.py',
                'ckpt':
                'textdet/'
                'panet/'
                'panet_resnet18_fpem-ffm_600e_ctw1500/'
                'panet_resnet18_fpem-ffm_600e_ctw1500_20220826_144818-980f32d0.pth'  # noqa: E501
            },
            'PANet_IC15': {
                'config': 'textdet/panet/panet_resnet18_fpem-ffm_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/panet_resnet18_fpem-ffm_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'PS_CTW': {
                'config':
                'textdet/'
                'psenet/psenet_resnet50_fpnf_600e_ctw1500.py',
                'ckpt':
                'textdet/'
                'psenet/'
                'psenet_resnet50_fpnf_600e_ctw1500/'
                'psenet_resnet50_fpnf_600e_ctw1500_20220825_221459-7f974ac8.pth'  # noqa: E501
            },
            'PS_IC15': {
                'config': 'textdet/psenet/psenet_resnet50_fpnf_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textdet/psenet_resnet50_fpnf_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'TextSnake': {
                'config':
                'textdet/'
                'textsnake/textsnake_resnet50_fpn-unet_1200e_ctw1500.py',
                'ckpt':
                'textdet/'
                'textsnake/'
                'textsnake_resnet50_fpn-unet_1200e_ctw1500/'
                'textsnake_resnet50_fpn-unet_1200e_ctw1500_20220825_221459-c0b6adc4.pth'  # noqa: E501
            },
            # Recognition models
            'CRNN': {
                'config': 'textrecog/crnn/crnn_mini-vgg_5e_lvdb.py',
                'ckpt':
                'textrecog/crnn/crnn_mini-vgg_5e_mj/crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth'  # noqa: E501
            },
            'SAR': {
                'config': 'textrecog/sar/sar_resnet31_parallel-decoder_5e_lvdb.py',
                'ckpt':
                'textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real_20220915_171910-04eb4e75.pth'  # noqa: E501
            },
            'NRTR_1/16-1/8': {
                'config':
                'textrecog/'
                'nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj.py',
                'ckpt':
                'textrecog/'
                'nrtr/nrtr_resnet31-1by16-1by8_6e_st_mj/nrtr_resnet31-1by16-1by8_6e_st_mj_20220920_143358-43767036.pth'  # noqa: E501
            },
            'NRTR_1/8-1/4': {
                'config':
                'textrecog/'
                'nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj.py',
                'ckpt':
                'textrecog/'
                'nrtr/nrtr_resnet31-1by8-1by4_6e_st_mj/nrtr_resnet31-1by8-1by4_6e_st_mj_20220916_103322-a6a2a123.pth'  # noqa: E501
            },
            'RobustScanner': {
                'config': 'textrecog/robust_scanner/robustscanner_resnet31_5e_lvdb.py',
                'ckpt':
                'textrecog/'
                'robust_scanner/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real/robustscanner_resnet31_5e_st-sub_mj-sub_sa_real_20220915_152447-7fc35929.pth'  # noqa: E501
            },
            'SATRN': {
                'config': 'textrecog/satrn/satrn_shallow_5e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textrecog/satrn_shallow_5e_lvdb_epoch_2.pth' # to be updated
            },
            'SATRN_sm': {
                'config': 'textrecog/satrn/satrn_shallow-small_5e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textrecog/satrn_shallow-small_5e_lvdb_epoch_2.pth' # to be updated
            },
            'ABINet': {
                'config': 'textrecog/abinet/abinet_20e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textrecog/abinet_20e_lvdb_epoch_20.pth'  # noqa: E501
            },
            'ABINet_Vision': {
                'config':
                'textrecog/abinet/abinet-vision_20e_st-an_mj.py',
                'ckpt':
                'textrecog/'
                'abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth'  # noqa: E501
            },
            'MASTER': {
                'config': 'textrecog/master/master_resnet31_5e_lvdb.py',
                'ckpt': 'https://sf.anismk.de/static/textrecog/master_resnet31_5e_lvdb_epoch_3.pth' # to be updated
            }
        }
