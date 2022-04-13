/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.testarmlapplication.classi

import android.util.Log
import com.google.mlkit.vision.common.PointF3D
import java.util.regex.Pattern


/**
 * Reads Pose samples from a csv file.
 */
class PoseSample(val name: String, val className: String, landmarks: List<PointF3D>) {
    val embedding: List<PointF3D> = PoseEmbedding.getPoseEmbedding(landmarks)

    companion object {
        private const val TAG = "PoseSample"
        private const val NUM_LANDMARKS = 33
        private const val NUM_DIMS = 3
        @JvmStatic
        fun getPoseSample(csvLine: String?, separator: String?): PoseSample? {
            val tokens: List<String> = Pattern.compile(separator).split(csvLine).toList()
            // Format is expected to be Name,Class,X1,Y1,Z1,X2,Y2,Z2...
            // + 2 is for Name & Class.
            if (tokens.size != NUM_LANDMARKS * NUM_DIMS + 2) {
                Log.e(TAG, "Invalid number of tokens for PoseSample")
                return null
            }
            val name = tokens[0]
            val className = tokens[1]
            val landmarks: MutableList<PointF3D> = ArrayList()
            // Read from the third token, first 2 tokens are name and class.
            var i = 2
            while (i < tokens.size) {
                try {
                    landmarks.add(
                        PointF3D.from(
                            tokens[i].toFloat(),
                            tokens[i + 1].toFloat(),
                            tokens[i + 2].toFloat()
                        )
                    )
                } catch (e: NullPointerException) {
                    Log.e(TAG, "Invalid value " + tokens[i] + " for landmark position.")
                    return null
                } catch (e: NumberFormatException) {
                    Log.e(TAG, "Invalid value " + tokens[i] + " for landmark position.")
                    return null
                }
                i += NUM_DIMS
            }
            return PoseSample(name, className, landmarks)
        }
    }

}