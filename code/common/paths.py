
import os

base_path = os.path.dirname(__file__)

class Paths:
    Results        = os.path.join(base_path, "../../data/results")
    Database       = os.path.join(base_path, "../../data/database")
    Segmentation   = os.path.join(base_path, "../../data/segmentation")
    Projects       = os.path.join(base_path, "../../data/labels")
    Models         = os.path.join(base_path, "../../data/models")
    #Images         = os.path.join(base_path, "../../data/images")
    Images         = os.path.join(base_path, "../../data")
    #TrainImages    = "../../data/train"
    #ValidImages    = "../../data/valid"
    Labels         = os.path.join(base_path, "../../data/labels")
    Data           = os.path.join(base_path, "../../data")
    Baseline       = os.path.join(base_path, "../../data/baseline")
    #Reference      = os.path.join(base_path, "../../data/reference")
    #Reference      = os.path.join(base_path, "../../data/eval2")
    Reference      = os.path.join(base_path, "../../data")
    Membranes      = '%s/reference/labels/membranes/test'%(Reference)
    ValidLabels    = '%s/reference/labels/validate'%(Reference)
    ValidMembranes = '%s/reference/labels/membranes/validate'%(Reference)
    ValidGray      = '%s/reference/images/validate'%(Reference)
    TestLabels     = '%s/labels/test'%(Reference)
    TestGrayscale  = '%s/test'%(Reference)
    TrainGrayscale = '%s/train'%(Reference)
    ValidGrayscale = '%s/valid'%(Reference)

