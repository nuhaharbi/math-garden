
var model;

async function loadModel() {
    model = await tf.loadGraphModel('TFJS/model.json');

}

function predictImage() {

    let image = cv.imread(canvas);
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(image, image, 175, 255, cv.THRESH_BINARY);

    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();

    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    image = image.roi(rect);

    var height = image.rows;
    var weidth = image.cols;

    if(height > weidth){
        height = 20;
        const scaleFactor = image.rows / height;
        weidth = Math.round(image.cols / scaleFactor);
    }else{
        weidth = 20;
        const scaleFactor = image.cols / weidth;
        height = Math.round(image.rows / scaleFactor);   
    }

    let newSize = new cv.Size(weidth, height);
    cv.resize(image, image, newSize, 0, 0, cv.INTER_AREA);


    const LEFT = Math.ceil(4 + (20 - weidth) / 2);
    const RIGHT = Math.floor(4 + (20 - weidth) / 2);
    const TOP = Math.ceil(4 + (20 - height) / 2);
    const BOTTOM = Math.floor(4 + (20 - height) / 2);
    let s = new cv.Scalar(0, 0, 0, 0);
    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT, cv.BORDER_CONSTANT, s);

    // Get the center of mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt, false);

    const cx = Moments.m10 / Moments.m00 ; 
    const cy = Moments.m01 / Moments.m00 ;
    

    // Shift the Image
    const X_SHIFT = Math.round(image.cols/2.0 - cx);
    const Y_SHIFT = Math.round(image.rows/2.0 - cy);
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    newSize = new cv.Size(image.cols, image.rows);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, s);
    
    //Normalies the pixels
    let pixelValues = image.data;

    pixelValues = Float32Array.from(pixelValues);

    pixelValues = pixelValues.map(function(item){
        return item / 255.0 ; 
    });

    // Create the tensor
    const X = tf.tensor([pixelValues]);
    

    // Predict 
    const result = model.predict(X);
    result.print();
    
    const output = result.dataSync()[0];

    // Cleanup
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();
    X.dispose();
    result.dispose();

    return output;
}