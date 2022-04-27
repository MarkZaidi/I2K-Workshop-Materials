/** Scripts enabling use of pretrained stardist models for nucleus segmentation on Brightfield or IF images
 * 3 pretrained models are available at https://github.com/stardist/stardist-imagej/tree/master/src/main/resources/models/2D
 * and must be downloaded prior to running this script. Furthermore, you need to build QuPath with tensorflow (verified with CPU)
 * using the instructions here: https://qupath.readthedocs.io/en/latest/docs/advanced/stardist.html
 * he_heavy_augment is a H&E-trained model, and requires a 3-channel input (like H&E or HDAB). dsb2018_paper and dsb2018_heavy_augment
 * are IF trained models, and requires a 1-channel input (either an IF nuclear marker like DAPI, or a deconvolved nuclear marker
 * from brightfield images like hematoxylin). This allows pretrained IF models to be used for both IF and brightfield segmentation
 */
//Variables to set *************************************************************
selectAnnotations();
def model_trained_on_single_channel=1 //Set to 1 if the pretrained model you're using was trained on IF sections, set to 0 if trained on brightfield
param_channel=22 //channel to use for nucleus detection. First channel in image is channel 1. If working with H&E or HDAB, channel 1 is hematoxylin.
param_median=0 //median preprocessing: Requires an int value corresponding to the radius of the median filter kernel. For radii larger than 2, the image must be in uint8 bit depth. Default 0
param_divide=1 //division preprocessing: int or floating point, divides selected channel intensity by value before segmenting. Useful when normalization is disabled. Default 1
param_add=0 //addition preprocessing: int or floating point, add value to selected channel intensity before segmenting. Useful when normalization is disabled. Default 0
param_threshold = 0.5//threshold for detection. All cells segmented by StarDist will have a detection probability associated with it, where higher values indicate more certain detections. Floating point, range is 0 to 1. Default 0.5
param_pixelsize=0.5 //Pixel scale to perform segmentation at. Set to 0 for image resolution (default). Int values accepted, greater values  will be faster but may yield poorer segmentations.
param_tilesize=1024 //size of tile in pixels for processing. Must be a multiple of 16. Lower values may solve any memory-related errors, but can take longer to process. Default is 1024.
param_expansion=10 //size of cell expansion in pixels. Default is 10.
def min_nuc_area=10 //remove any nuclei with an area less than or equal to this value
nuc_area_measurement='Nucleus: Area µm^2'
def min_nuc_intensity=0 //remove any detections with an intensity less than or equal to this value
nuc_intensity_measurement='Hematoxylin: Nucleus: Mean'
normalize_low_pct=1 //lower limit for normalization. Set to 0 to disable
normalize_high_pct=99 // upper limit for normalization. Set to 100 to disable.

// Specify the model directory (you will need to change this!). Uncomment the model you wish to use
//Brightfield models
    //def pathModel = 'C:/Users/Mark Zaidi/Documents/QuPath/Stardist Trained Models/he_heavy_augment'
// IF models
    //def pathModel = 'C:/Users/Mark Zaidi/Documents/QuPath/Stardist Trained Models/dsb2018_paper'
    def pathModel = 'C:/Users/Mark Zaidi/Documents/QuPath/Stardist Trained Models/dsb2018_heavy_augment'

//End of variables to set ******************************************************
param_channel=param_channel-1 // corrects for off-by-one error
normalize_high_pct=normalize_high_pct-0.000000000001 //corrects for some bizarre normalization issue when attempting to set 100 as the upper limit

// Import plugins
//import qupath.tensorflow.stardist.StarDist2D
import qupath.ext.stardist.StarDist2D
import static qupath.lib.gui.scripting.QPEx.*
import groovy.time.*
//

// Specify whether the above model was trained using a single-channel image (e.g. IF DAPI)
// Get current image - assumed to have color deconvolution stains set
def imageData = getCurrentImageData()
def isBrightfield=imageData.isBrightfield()
def stains = imageData.getColorDeconvolutionStains() //will be null if IF

if (model_trained_on_single_channel!=1 && isBrightfield==false) {
    // If brightfield model but fluorescent image
    throw new Exception("Cannot use brightfield trained model to segment nuclei on IF image")
} else if (model_trained_on_single_channel == 1 && isBrightfield==true){
    //If fluorescent model but brightfield image (use deconvolution)
    println 'Performing detection on Brightfield image using single-channel trained model'
     stardist = StarDist2D.builder(pathModel)
            .preprocess(
                    ImageOps.Channels.deconvolve(stains),
                    ImageOps.Channels.extract(param_channel),
                    ImageOps.Filters.median(param_median),
                    ImageOps.Core.divide(param_divide),
                    ImageOps.Core.add(param_add)
            ) // Optional preprocessing (can chain multiple ops)

            .threshold(param_threshold)              // Prediction threshold
            .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization
            .pixelSize(param_pixelsize)              // Resolution for detection
            .doLog()
            .includeProbability(true)
            .measureIntensity()
            .tileSize(param_tilesize)
            .measureShape()
            .cellExpansion(param_expansion) //Cell expansion in microns
            .constrainToParent(false)


             .build()
} else if (model_trained_on_single_channel == 1 && isBrightfield==false){
    //If IF model and IF image (no deconvolution preprocessing)
    println 'Performing detection on single channel image using single-channel trained model'

    stardist = StarDist2D.builder(pathModel)
            .preprocess(
                    ImageOps.Channels.extract(param_channel),
                    ImageOps.Filters.median(param_median),
                    ImageOps.Core.divide(param_divide),
                    ImageOps.Core.add(param_add)
            ) // Optional preprocessing (can chain multiple ops)

            .threshold(param_threshold)              // Prediction threshold
            .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization. REQUIRED FOR IMC DATA
            .pixelSize(param_pixelsize)              // Resolution for detection
            .doLog()
            .includeProbability(true)
            .measureIntensity()
            .tileSize(param_tilesize)
            .measureShape()
            .cellExpansion(param_expansion)
            .constrainToParent(false)
            .build()
} else if (model_trained_on_single_channel == 0 && isBrightfield==true) {
    // If brightfield model and brightfield image
    println 'Performing detection on brightfield image using brightfield trained model'

    stardist = StarDist2D.builder(pathModel)
            .preprocess(
                    ImageOps.Filters.median(param_median),
                    ImageOps.Core.divide(param_divide),
                    ImageOps.Core.add(param_add)
            ) // Optional preprocessing (can chain multiple ops)

            .threshold(param_threshold)              // Prediction threshold
            .normalizePercentiles(normalize_low_pct,normalize_high_pct) // Percentile normalization. REQUIRED FOR IMC DATA
            .pixelSize(param_pixelsize)              // Resolution for detection
            .doLog()
            .includeProbability(true)
            .measureIntensity()
            .tileSize(param_tilesize)
            .measureShape()
            .cellExpansion(param_expansion)
            .constrainToParent(false)
            .build()

}
//Run stardist in selected annotation
def pathObjects = getSelectedObjects()
print(pathObjects)
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("StarDist", "Please select a parent object!")
    return
}
//clearDetections()
def timeStart_CellDetection = new Date()
stardist.detectObjects(imageData, pathObjects)
TimeDuration CellDetection_duration = TimeCategory.minus(new Date(), timeStart_CellDetection)

//filter out small and low intensity nuclei


def toDelete = getDetectionObjects().findAll {measurement(it, nuc_area_measurement) <= min_nuc_area}
removeObjects(toDelete, true)
def toDelete2 = getDetectionObjects().findAll {measurement(it, nuc_intensity_measurement) <= min_nuc_intensity}
removeObjects(toDelete2, true)

println ('Done in ' + CellDetection_duration)
runObjectClassifier("composite_classifer_v1");