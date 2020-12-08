package com.tensorflow.android.spleetermobileapp

import android.os.Bundle
import android.os.Environment
import androidx.appcompat.app.AppCompatActivity
import com.jlibrosa.audio.JLibrosa
import org.apache.commons.math3.complex.Complex
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.util.stream.IntStream


class MainActivity : AppCompatActivity() {

    var jLibrosa: JLibrosa? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val externalStorage: File = Environment.getExternalStorageDirectory()

        val audioFilePath = externalStorage.absolutePath + "/images/AClassicEducation.wav";

        val defaultSampleRate = -1 //-1 value implies the method to use default sample rate

        val defaultAudioDuration =
            20 //-1 value implies the method to process complete audio duration


        jLibrosa = JLibrosa()

        val stereoFeatureValues =
            jLibrosa!!.loadAndReadStereo(audioFilePath, defaultSampleRate, defaultAudioDuration)

        val stereoTransposeFeatValues: Array<FloatArray> =
            transposeMatrix(stereoFeatureValues)

        val stftValues: Array<Array<Array<Array<Float?>>>> =
            _stft(stereoTransposeFeatValues)

        val tflite: Interpreter

        //load the TFLite model in 'MappedByteBuffer' format using TF Interpreter
        val tfliteModel: MappedByteBuffer =  FileUtil.loadMappedFile(applicationContext, "model.tflite")





        /** Options for configuring the Interpreter.  */
        val tfliteOptions =
            Interpreter.Options()
        tfliteOptions.setNumThreads(2)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        //get the datatype and shape of the input tensor to be fed to tflite model
        val imageTensorIndex = 0

        val imageDataType: DataType = tflite.getInputTensor(imageTensorIndex).dataType()

        val imageDataShape: IntArray = tflite.getInputTensor(imageTensorIndex).shape()

        //get the datatype and shape of the output prediction tensor from tflite model
        val probabilityTensorIndex = 0
        val probabilityShape =
            tflite.getOutputTensor(probabilityTensorIndex).shape()
        val probabilityDataType: DataType =
            tflite.getOutputTensor(probabilityTensorIndex).dataType()

        var byteBuffer : ByteBuffer = ByteBuffer.allocate(4*stftValues.size*stftValues[0].size)

        for(i in 0 until stftValues.size){
            val valArray: Array<Array<Array<Float?>>> = stftValues[i]
            val inpShapeDim: IntArray = intArrayOf(1,1,stftValues[0].size,1)
            val valInTnsrBuffer: TensorBuffer = TensorBuffer.createDynamic(imageDataType)
            valInTnsrBuffer.loadArray(valArray, inpShapeDim)
            val valInBuffer : ByteBuffer = valInTnsrBuffer.getBuffer()
            byteBuffer.put(valInBuffer)
        }

        byteBuffer.rewind()

        //val inpBuffer: ByteBuffer? = convertBitmapToByteBuffer(bitmp)
        val outputTensorBuffer: TensorBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
        //run the predictions with input and output buffer tensors to get probability values across the labels
        tflite.run(byteBuffer, outputTensorBuffer.getBuffer())

        print(10000)
    }

    private fun _stft(stereoMatrix: Array<FloatArray>): Array<Array<Array<Array<Float?>>>> {
        val N = 4096
        val H = 1024
        val sampleRate = 44100
        val stftValuesList =
            ArrayList<Array<Array<Complex?>>>()
        for (i in 0 until stereoMatrix[0].size) {
            val doubleStream: FloatArray = getColumnFromMatrix(stereoMatrix, i)
            val stftComplexValues =
                jLibrosa!!.generateSTFTFeatures(doubleStream, sampleRate, 40, 4096, 128, 1024)
            val transposedSTFTComplexValues: Array<Array<Complex?>> =
                transposeMatrix(stftComplexValues)
            stftValuesList.add(transposedSTFTComplexValues)
        }
        val segmentLen = 512
        val stft3DMatrixValues: Array<Array<Array<Double?>>> =
            gen3DMatrixFrom2D(stftValuesList, segmentLen)
        val splitValue = (stft3DMatrixValues.size + segmentLen - 1) / segmentLen
        return gen4DMatrixFrom3D(stft3DMatrixValues, splitValue, segmentLen)
    }


    private fun gen4DMatrixFrom3D(
        stft3DMatrixValues: Array<Array<Array<Double?>>>,
        splitValue: Int,
        segmentLen: Int
    ): Array<Array<Array<Array<Float?>>>> {
        val yVal = 1024
        val zVal: Int = stft3DMatrixValues[0][0].size
        val stft4DMatrixValues =
            Array(
                splitValue
            ) {
                Array(
                    segmentLen
                ) {
                    Array(
                        yVal
                    ) { arrayOfNulls<Float?>(zVal) }
                }
            }
        for (p in 0 until splitValue) {
            for (q in 0 until segmentLen) {
                val retInd = p * segmentLen + q
                for (r in 0 until yVal) {
                    for (s in 0 until zVal) {
                        stft4DMatrixValues[p][q][r][s] =
                            stft3DMatrixValues[retInd][r][s]?.toFloat()
                    }
                }
            }
        }
        return stft4DMatrixValues
    }


    private fun gen3DMatrixFrom2D(
        mat2DValuesList: ArrayList<Array<Array<Complex?>>>,
        segmentLen: Int
    ): Array<Array<Array<Double?>>> {
        val padSize: Int = computePadSize(mat2DValuesList[0].size, segmentLen)
        val matrixXLen: Int = mat2DValuesList[0].size + padSize
        val complex3DMatrix =
            Array(
                matrixXLen
            ) {
                Array(
                    mat2DValuesList[0][0].size
                ) { arrayOfNulls<Double>(mat2DValuesList.size) }
            }
        for (k in 0 until mat2DValuesList.size) {
            val mat2DValues = mat2DValuesList[k]
            for (i in 0 until matrixXLen) {
                for (j in 0 until mat2DValues[0].size) {
                    var value : Double? = 0.0
                    if (i < mat2DValues.size) {
                        value = mat2DValues[i][j]?.abs()
                    }
                    complex3DMatrix[i][j][k] = value
                }
            }
        }
        return complex3DMatrix
    }


    private fun computePadSize(currentMatrixLen: Int, segmentLen: Int): Int {
        val tensorSize = currentMatrixLen % segmentLen
        return segmentLen - tensorSize
    }

    private fun getColumnFromMatrix(
        floatMatrix: Array<FloatArray>,
        column: Int
    ): FloatArray {

        val doubleStream: DoubleArray = IntStream.range(0, floatMatrix.size)
            .mapToDouble({ i -> floatMatrix[i][column].toDouble()}).toArray()
        val floatArray = FloatArray(doubleStream.size)
        for (i in doubleStream.indices) {
            floatArray[i] = doubleStream[i].toFloat()
        }
        return floatArray
    }


    fun transposeMatrix(matrix: Array<Array<Complex?>>): Array<Array<Complex?>> {
        val m = matrix.size
        val n: Int = matrix[0].size
        val transposedMatrix =
            Array(n) { arrayOfNulls<Complex>(m) }
        for (x in 0 until n) {
            for (y in 0 until m) {
                transposedMatrix[x][y] = matrix[y][x]
            }
        }
        return transposedMatrix
    }


    fun transposeMatrix(matrix: Array<FloatArray>): Array<FloatArray> {
        val m = matrix.size
        val n: Int = matrix[0].size
        val transposedMatrix =
            Array(n) { FloatArray(m) }
        for (x in 0 until n) {
            for (y in 0 until m) {
                transposedMatrix[x][y] = matrix[y][x]
            }
        }
        return transposedMatrix
    }

}
