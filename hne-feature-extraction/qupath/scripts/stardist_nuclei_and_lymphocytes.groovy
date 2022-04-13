import qupath.tensorflow.stardist.StarDist2D
import qupath.lib.io.GsonTools
import static qupath.lib.gui.scripting.QPEx.*
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.60968 0.65246 0.4501 ", "Stain 2" : "Eosin", "Values 2" : "0.21306 0.87722 0.43022 ", "Background" : " 243 243 243 "}');

def imageData = getCurrentImageData()
def server = imageData.getServer()

// get dimensions of slide
minX = 0
minY = 0
maxX = server.getWidth()
maxY = server.getHeight()

print 'maxX' + maxX
print 'maxY' + maxY

// create rectangle roi (over entire area of image) for detections to be run over
def plane = ImagePlane.getPlane(0, 0)
def roi = ROIs.createRectangleROI(minX, minY, maxX-minX, maxY-minY, plane)
def annotationROI = PathObjects.createAnnotationObject(roi)
addObject(annotationROI)
selectAnnotations();
def pathModel = '/models/he_heavy_augment'
def cell_expansion_factor = 3.0
def cellConstrainScale = 1.0
def stardist = StarDist2D.builder(pathModel)
        .threshold(0.5)              // Probability (detection) threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(0.5)              // Resolution for detection
        .cellExpansion(cell_expansion_factor)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(cellConstrainScale)     // Constrain cell expansion using nucleus size
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .nThreads(10)
        .build()
// select rectangle object created
selectObjects {
   //Some criteria here
   return it == annotationROI
}
def pathObjects = getSelectedObjects()
print 'Selected ' + pathObjects.size()
// stardist segmentations
stardist.detectObjects(imageData, pathObjects)
def celldetections = getDetectionObjects()
print 'Detected' + celldetections.size()
selectDetections();
// obj classifier
runObjectClassifier("/models/ANN_StardistSeg3.0CellExp1.0CellConstraint_AllFeatures_LymphClassifier.json")
def filename = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
saveDetectionMeasurements('/data/results/' + filename + '.tsv')
