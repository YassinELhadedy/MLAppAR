package com.example.testarmlapplication.ml.classification

import android.app.Activity
import android.media.Image
import com.example.testarmlapplication.ml.classification.utils.ImageUtils
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions
import kotlinx.coroutines.tasks.asDeferred

class MLKitPoserDetector(context: Activity) : PoseDetector(context) {



    val options = AccuratePoseDetectorOptions.Builder()
        .setDetectorMode(AccuratePoseDetectorOptions.SINGLE_IMAGE_MODE)
        .build()
    val detector = PoseDetection.getClient(options)


    override suspend fun analyze(image: Image, imageRotation: Int): List<PoseDetectorObjectResult> {
        // `image` is in YUV (https://developers.google.com/ar/reference/java/com/google/ar/core/Frame#acquireCameraImage()),
        val convertYuv = convertYuv(image)

        // The model performs best on upright images, so rotate it.
        val rotatedImage = ImageUtils.rotateBitmap(convertYuv, imageRotation)

        val inputImage = InputImage.fromBitmap(rotatedImage, 0)

        val mlKitPoseDetectedObjects: Pose = detector.process(inputImage).asDeferred().await()
        return mlKitPoseDetectedObjects.allPoseLandmarks.mapNotNull { obj ->
//            val bestLabel = obj.labels.maxByOrNull { label -> label.confidence } ?: return@mapNotNull null
//            val coords = obj.boundingBox.exactCenterX().toInt() to obj.boundingBox.exactCenterY().toInt()
//            val rotatedCoordinates = coords.rotateCoordinates(rotatedImage.width, rotatedImage.height, imageRotation)
            PoseDetectorObjectResult(obj.inFrameLikelihood, obj.landmarkType.toString(), Pair(obj.position3D.x.toInt(),obj.position3D.y.toInt()))
        }
    }


}