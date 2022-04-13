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

import java.util.Collections
import java.util.HashMap

/**
 * Represents Pose classification result as outputted by [PoseClassifier]. Can be manipulated.
 */
class ClassificationResult {
    // For an entry in this map, the key is the class name, and the value is how many times this class
    // appears in the top K nearest neighbors. The value is in range [0, K] and could be a float after
    // EMA smoothing. We use this number to represent the confidence of a pose being in this class.
    private val classConfidences: MutableMap<String, Float>
    val allClasses: Set<String>
        get() = classConfidences.keys

    fun getClassConfidence(className: String): Float {
        return if (classConfidences.containsKey(className)) classConfidences[className]!! else 0F
    }

    val maxConfidenceClass: String
        get() = Collections.max<Map.Entry<String, Float>>(
            classConfidences.entries
        ) { (_, value), (_, value1) -> (value - value1).toInt() }
            .key

    fun incrementClassConfidence(className: String) {
        classConfidences[className] =
            (if (classConfidences.containsKey(className)) classConfidences[className]!! + 1 else 1) as Float
    }

    fun putClassConfidence(className: String, confidence: Float) {
        classConfidences[className] = confidence
    }

    init {
        classConfidences = HashMap()
    }
}