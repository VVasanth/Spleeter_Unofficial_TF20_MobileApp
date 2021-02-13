package com.tensorflow.android.spleetermobileapp

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.os.Process
import android.text.TextUtils
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.arthenica.mobileffmpeg.Config
import com.arthenica.mobileffmpeg.FFmpeg
import com.jlibrosa.audio.JLibrosa
import com.tensorflow.android.R
import kotlinx.android.synthetic.main.activity_main.*
import org.apache.commons.math3.complex.Complex
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.util.stream.IntStream
import kotlin.math.pow


class MainActivity : AppCompatActivity() {

    var jLibrosa: JLibrosa? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val externalStorage: File = Environment.getExternalStorageDirectory()

        //val audioFilePath_1 = externalStorage.absolutePath + "/images/AClassicEducation.wav";

        val audioFilePath = externalStorage.absolutePath + "/audio-separator-input/";

        val fileNames: MutableList<String> = ArrayList()


        File(audioFilePath).walk().forEach{

            if(it.absolutePath.endsWith(".wav")){
                fileNames.add(it.name)
            }

        }

        // access the spinner
        val spinner = findViewById<Spinner>(R.id.spinner)
        if (spinner != null) {
            val adapter = ArrayAdapter(this,
                android.R.layout.simple_spinner_item, fileNames)
            spinner.adapter = adapter

        }

        process_button.setOnClickListener( View.OnClickListener {
            val selFilePath = spinner.selectedItem.toString()
            var audioFilePath = audioFilePath + '/' + selFilePath;
            if ( !TextUtils.isEmpty( selFilePath ) ){
                val intent = Intent(this, ListActivity::class.java)
                startActivity(intent)
                processAudioSeparation(audioFilePath, applicationContext)
                Toast.makeText( this@MainActivity, "Audio file is being processed. Pls wait for ~minutes for the process to complete.", Toast.LENGTH_LONG).show();
            }
            else{
                Toast.makeText( this@MainActivity, "Please select the file and click on the process button.", Toast.LENGTH_LONG).show();
            }
        })

        view_button.setOnClickListener( View.OnClickListener {
                val intent = Intent(this, ListActivity::class.java)
                startActivity(intent)
        })

        print(10000)
    }


    private fun processAudioSeparation(audioFilePath: String, context: Context) {

        val loadRunnable = Runnable {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)

            val defaultSampleRate = -1 //-1 value implies the method to use default sample rate

            val defaultAudioDuration =
                15 //-1 value implies the method to process complete audio duration

            val audioFileFullName = audioFilePath.substringAfterLast("/")
            val audioFileName = audioFileFullName.substringBeforeLast(".")

            jLibrosa = JLibrosa()

            val stereoFeatureValues =
                jLibrosa!!.loadAndReadStereo(audioFilePath, defaultSampleRate, defaultAudioDuration)


            val stereoTransposeFeatValues: Array<FloatArray> =
                transposeMatrix(stereoFeatureValues)

            val stftValueList: ArrayList<Array<Array<Complex?>>> = _stft(stereoTransposeFeatValues)

            val stftValues: Array<Array<Array<Array<Float?>>>> = processSTFTValues(stftValueList)

            val modelNameList = arrayOf("vocals_model.tflite")

            var predictionOutputList : MutableList<MutableList<FloatArray>> = ArrayList()

            val consInstValuesStereoList:MutableList<MutableList<MutableList<FloatArray>>> = ArrayList()

            for(i in 0 until stftValues.size){
                val valArray: Array<Array<Array<Float?>>> = stftValues[i]

                /*
                var byteBuffer : ByteBuffer = ByteBuffer.allocate(4*valArray.size*valArray[0].size*valArray[0][0].size)

                for (j in 0 until valArray.size){
                    val valFloatArray: FloatArray = genFloatArray(valArray[j])
                    val inpShapeDim: IntArray = intArrayOf(1,1,valArray[0].size,2)
                    val valInTnsrBuffer: TensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32)
                    valInTnsrBuffer.loadArray(valFloatArray, inpShapeDim)
                    val valInBuffer : ByteBuffer = valInTnsrBuffer.getBuffer()
                    byteBuffer.put(valInBuffer)
                }
                byteBuffer.rewind()
                 */

                val inputInference =
                    Array(
                        1
                    ) {
                        Array(
                            512
                        ) { Array(1024) { FloatArray(2) } }
                    }

                for (i in 0 until 512) {
                    for (j in 0 until 1024) {
                        for (k in 0 until 2) {
                            inputInference[0][i][j][k] =
                                valArray[i][j][k]!!
                        }
                    }
                }


                var predictionStemOutputList : MutableList<Array<Array<Array<FloatArray>>>> = ArrayList()

                for(m in 0 until modelNameList.size) {
                    val matrixResultOutput =
                        executePredictionsFromTFLiteModelAsArray(modelNameList[m], inputInference)
                    predictionStemOutputList.add(matrixResultOutput)
                }
                val maskedResult:MutableList<Array<Array<Array<Complex>>>> = maskOutput(predictionStemOutputList, stftValueList)


                val insValuesStereoList:MutableList<MutableList<FloatArray>> = extractISTFT(maskedResult)

                consInstValuesStereoList.add(insValuesStereoList)
            }

            val procInstValuesStereoList:MutableList<MutableList<FloatArray>> = processConsolidatedInstValuesList(consInstValuesStereoList)

            for(p in 0 until procInstValuesStereoList.size){
                saveWavFromMagValues(procInstValuesStereoList[p],p,audioFileName)
            }

        }

        val processThread = Thread(loadRunnable);
        processThread.start();


    }

    private fun processConsolidatedInstValuesList(consInstValuesList: MutableList<MutableList<MutableList<FloatArray>>>):MutableList<MutableList<FloatArray>>{

        val segmentSize = consInstValuesList.size
        val instrumentSize = consInstValuesList[0].size
        val channelSize = consInstValuesList[0][0].size
        val magSize = consInstValuesList[0][0][0].size

        val procConsInstValuesList: MutableList<MutableList<FloatArray>> = ArrayList()

        for(j in 0 until instrumentSize){
            val channelValuesList: MutableList<FloatArray> = ArrayList()
            for(k in 0 until channelSize){
                val procFloatArrayList:ArrayList<Float> = ArrayList()
                for(l in 0 until magSize){
                    for (i in 0 until segmentSize){
                            procFloatArrayList.add(consInstValuesList[i][j][k][l])
                        }
                }
                channelValuesList.add(procFloatArrayList.toFloatArray())
            }
            procConsInstValuesList.add(channelValuesList)
        }

        return procConsInstValuesList
    }


    private fun saveWavFromMagValues(instrumentMagValues:MutableList<FloatArray>, fileSuffix:Int, audioInputFileName:String) {

        val byteArray = convertFloatArrayToByteArray(instrumentMagValues)

        val externalStorage_1: File = Environment.getExternalStorageDirectory()

        val audioFileName: String = audioInputFileName + "_vocal"

        val outputPath_1 = externalStorage_1.absolutePath + "/audio-separator-output/" + audioFileName + ".wav"

        val sampleOutputFile = externalStorage_1.absolutePath + "/audio-separator-output/bytes_j.txt"

        val pipe1: String = Config.registerNewFFmpegPipe(applicationContext)


        try {
            val file = File(sampleOutputFile)
            // Initialize a pointer
            // in file using OutputStream
            val os: OutputStream = FileOutputStream(file)

            // Starts writing the bytes in it
            os.write(byteArray)
            println(
                "Successfully"
                        + " byte inserted"
            )

            // Close the file
            os.close()
        } catch (e: Exception) {
            println("Exception: $e")
        }

        val ffmpegCommand = "-f f32le -ac 2 -ar 44100 -i " + pipe1 + " -b:a 128k -ar 44100 -strict -2 " + outputPath_1 + " -y"

        Runtime.getRuntime().exec(arrayOf("sh", "-c", "cat " + sampleOutputFile + " > " + pipe1))

        FFmpeg.execute(ffmpegCommand, " ");

    }

    private fun convertFloatArrayToByteArray(stereoArray: MutableList<FloatArray>): ByteArray? {

        val array: FloatArray = stereoArray[0]
        val n_channels = 2

        val consByteArray = ByteArray(4 * array.size * n_channels)

        for (i in array.indices) {
            for (j in stereoArray.indices) {

                val byteArray =
                    ByteBuffer.allocate(4).putFloat(stereoArray[j][i]).array()
                val leByteArray = convertBigEndianToLittleEndian(byteArray)
                for (k in leByteArray!!.indices) {
                    consByteArray[i * 8 + j*4 + k] = leByteArray[k]
                }
            }
        }

      return consByteArray
    }


    private fun convertBigEndianToLittleEndian(value: ByteArray): ByteArray? {
        val length = value.size
        val res = ByteArray(length)
        for (i in 0 until length) {
            res[length - i - 1] = value[i]
        }
        return res
    }

    private fun writeArrayToFile(arr2DValue: Array<Array<Complex>>){
        val builder = StringBuilder()
        for (i in 0 until arr2DValue.size)  //for each row
        {
            for (j in 0 until arr2DValue[0].size)  //for each column
            {
                builder.append(arr2DValue.get(i).get(j).toString() + "") //append to the output string
                if (j < arr2DValue[0].size - 1) //if this is not the last row element
                    builder.append(",") //then add comma (if you don't like commas you can use spaces)
            }
            builder.append("\n") //append new line at the end of the row
        }

        val externalStorage_1: File = Environment.getExternalStorageDirectory()

        val sampleOutputFile = externalStorage_1.absolutePath + "/images/twodarray.txt"

        val writer =
            BufferedWriter(FileWriter(sampleOutputFile))
        writer.write(builder.toString()) //save the string representation of the board

        writer.close()
    }

    private fun extractISTFT(maskedResultValueList: MutableList<Array<Array<Array<Complex>>>>): MutableList<MutableList<FloatArray>>{


        val insValuesStereoArrayList: MutableList<MutableList<FloatArray>> = ArrayList()

        for(i in 0 until maskedResultValueList.size){
            val maskedResultValue: Array<Array<Array<Complex>>> = maskedResultValueList[i]
            val magValuesFloatArrayList: MutableList<FloatArray> = ArrayList()

            for(p in 0 until maskedResultValue[0][0].size){
                val maskedResultValInstrument:Array<Array<Complex>> = Array(
                    maskedResultValue[0].size
                ) {
                    Array(
                        maskedResultValue.size
                    ) {Complex(0.0,0.0)}
                }
                for(q in 0 until maskedResultValue[0].size){
                    for(r in 0 until maskedResultValue.size){

                        maskedResultValInstrument[q][r] = maskedResultValue[r][q][p]

                    }
                }

                //writeArrayToFile(maskedResultValInstrument)

                val magValues:FloatArray = jLibrosa!!.generateInvSTFTFeatures(maskedResultValInstrument,
                    jLibrosa!!.sampleRate, 40,4096, 128, 1024)


                magValuesFloatArrayList.add(magValues)
            }
            insValuesStereoArrayList.add(magValuesFloatArrayList)
        }
        return insValuesStereoArrayList
    }


    private fun executePredictionsFromTFLiteModelAsArray(modelName:String, inputInference: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>> {

        val tflite: Interpreter

        //load the TFLite model in 'MappedByteBuffer' format using TF Interpreter
        val tfliteModel: MappedByteBuffer =  FileUtil.loadMappedFile(applicationContext, modelName)

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


        val outputTensorBuffer: TensorBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        val outputInference =
            Array(
                1
            ) {
                Array(
                    512
                ) { Array(1024) { FloatArray(2) } }
            }

        //run the predictions with input and output buffer tensors to get probability values across the labels
        tflite.run(inputInference, outputInference)

        return outputInference
    }




    private fun executePredictionsFromTFLiteModel(modelName:String, inpByteBuffer: ByteBuffer): Array<Array<Array<FloatArray>>> {

        val tflite: Interpreter

        //load the TFLite model in 'MappedByteBuffer' format using TF Interpreter
        val tfliteModel: MappedByteBuffer =  FileUtil.loadMappedFile(applicationContext, modelName)

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


        val outputTensorBuffer: TensorBuffer =
            TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
        //run the predictions with input and output buffer tensors to get probability values across the labels
        tflite.run(inpByteBuffer, outputTensorBuffer.getBuffer())
        val outputProcessor: TensorProcessor = TensorProcessor.Builder()
            .build()
        var modelOutputResult : TensorBuffer = outputProcessor.process(outputTensorBuffer)
        var matrixResultOutput : Array<Array<Array<FloatArray>>> = gen4DMatrixFromModelOutput(modelOutputResult, probabilityShape)
        return matrixResultOutput

    }

    private fun maskOutput(matrixResultOutputList: MutableList<Array<Array<Array<FloatArray>>>>, stftValues: ArrayList<Array<Array<Complex?>>>): MutableList<Array<Array<Array<Complex>>>> {

        val separationComponent: Int = 2
        val eps = 1e-10F
        val lenStems = 1 //2

        val pInd = matrixResultOutputList[0].size
        val qInd = matrixResultOutputList[0][0].size
        val rInd = matrixResultOutputList[0][0][0].size
        val sInd = matrixResultOutputList[0][0][0][0].size




        var outputSumMatrix: Array<Array<Array<FloatArray>>> =
            Array(
                pInd
            ) {
                Array(
                    qInd
                ) { Array(rInd) { FloatArray(sInd) } }
            }

            for (i in 0 until pInd) {
                for (j in 0 until qInd) {
                    for (k in 0 until rInd) {
                        for (l in 0 until sInd) {
                            outputSumMatrix[i][j][k][l] = 0.0F
                            for(x in 0 until matrixResultOutputList.size) {
                                outputSumMatrix[i][j][k][l] = outputSumMatrix[i][j][k][l] + matrixResultOutputList[x][i][j][k][l].pow(separationComponent).toFloat()
                            }
                            outputSumMatrix[i][j][k][l] = outputSumMatrix[i][j][k][l] + eps
                    }
                }
            }
        }


        var processedPredOutputList : MutableList<Array<Array<Array<Complex>>>> = ArrayList()

        for(x in 0 until matrixResultOutputList.size){

            val rInd_extn = 2049

            var procPredMatrix: Array<Array<Array<FloatArray>>> =
                Array(
                    pInd
                ) {
                    Array(
                        qInd
                    ) { Array(rInd_extn) { FloatArray(sInd) } }
                }

            for (i in 0 until pInd){
                for(j in 0 until qInd){
                    for (k in 0 until rInd){
                        for (l in 0 until sInd){
                            var value = matrixResultOutputList[x][i][j][k][l]
                            var procValue = ((value.pow(separationComponent)) + (eps/lenStems)) /// outputSumMatrix[i][j][k][l]
                            procPredMatrix[i][j][k][l] = procValue
                        }
                    }
                }
            }


            var procPred3DMatrix: Array<Array<Array<Complex>>> = gen3DMatrixFrom4DMatrix(procPredMatrix, stftValues)

            processedPredOutputList.add(procPred3DMatrix)

        }

        return processedPredOutputList

    }

    private fun gen3DMatrixFrom4DMatrix(procPredMatrix: Array<Array<Array<FloatArray>>>, stftValues: ArrayList<Array<Array<Complex?>>>): Array<Array<Array<Complex>>> {

        val pInd = procPredMatrix.size
        val qInd = procPredMatrix[0].size
        val rInd = procPredMatrix[0][0].size
        val sInd = procPredMatrix[0][0][0].size

        val stftVal1DSize: Int = stftValues[0].size

        var procPred3DMatrix: Array<Array<Array<Complex>>> =
            Array(
                stftVal1DSize
            ) { Array(procPredMatrix[0][0].size) { Array(procPredMatrix[0][0][0].size){
                Complex(0.0,0.0)
            } } }

        var breakFlag: Int = 0

        for(p in 0 until pInd){
            for(q in 0 until qInd){
                for(r in 0 until rInd){
                    for(s in 0 until sInd){
                        val pqInd = (p * q) + q
                        if(pqInd<stftVal1DSize){
                            val dblVal : Double = procPredMatrix[p][q][r][s].toDouble()
                            val complxVal: Complex = Complex(dblVal)
                            val stftIndexVal : Complex? = stftValues[s][pqInd][r]
                            procPred3DMatrix[pqInd][r][s] = complxVal.multiply(stftIndexVal)
                        }else{
                            breakFlag = 1
                            break
                        }
                    }
                    if(breakFlag==1){
                        break
                    }

                }

                if(breakFlag==1){
                    break
                }
            }
                if(breakFlag==1){
                    break
                }
        }

        return procPred3DMatrix
    }

    private fun multiplyComplexNumbers(complxVal1: Complex, complxVal2: Complex): Complex {
        val realVal : Double = (complxVal1.real * complxVal2.real) - (complxVal1.imaginary * complxVal2.imaginary)
        val imgVal : Double = (complxVal1.real * complxVal2.imaginary) + (complxVal1.imaginary * complxVal2.real)
        return Complex(realVal, imgVal)
    }

    private fun gen4DMatrixFromModelOutput(valTensorBuffer:TensorBuffer, outputShape:IntArray): Array<Array<Array<FloatArray>>> {

        var outputMatrixValue: Array<Array<Array<FloatArray>>> =
            Array(
                outputShape[0]
            ) {
                Array(
                    outputShape[1]
                ) { Array(outputShape[2]) { FloatArray(outputShape[3]) } }
            }

        for (i in 0 until outputShape[0]){
            for(j in 0 until outputShape[1]){
                for(k in 0 until outputShape[2]){
                    for(l in 0 until outputShape[3]){
                        val indexVal = i*outputShape[1] + j*outputShape[2] + k* outputShape[3] + l
                        outputMatrixValue[i][j][k][l] = valTensorBuffer.getFloatValue(indexVal)
                    }
                }
            }
        }

        return outputMatrixValue

    }



    private fun genFloatArray(valInpArray:Array<Array<Float?>>): FloatArray{
        val arrIndex1 = valInpArray.size
        val arrIndex2 = valInpArray[0].size

        val floatArraySize = arrIndex1 * arrIndex2

        val valFloatArray : FloatArray = FloatArray(floatArraySize)

        for (i in 0 until arrIndex1){
            for(j in 0 until arrIndex2){
                    val arrIndexVal = (i * arrIndex2) + j
                    valFloatArray[arrIndexVal] = valInpArray[i][j]!!
            }
        }
        return valFloatArray
    }

    private fun _stft(stereoMatrix: Array<FloatArray>): ArrayList<Array<Array<Complex?>>> {
        val N = 4096
        val H = 1024
        val sampleRate = 44100
        val stftValuesList =
            ArrayList<Array<Array<Complex?>>>()
        for (i in 0 until stereoMatrix[0].size) {
            val doubleStream: FloatArray = getColumnFromMatrix(stereoMatrix, i)
            val stftComplexValues =
                jLibrosa!!.generateSTFTFeaturesWithPadOption(doubleStream, sampleRate, 40, 4096, 128, 1024, false)
            val transposedSTFTComplexValues: Array<Array<Complex?>> =
                transposeMatrix(stftComplexValues)
            stftValuesList.add(transposedSTFTComplexValues)
        }
        return stftValuesList
    }


    private fun processSTFTValues(stftValuesList: ArrayList<Array<Array<Complex?>>>): Array<Array<Array<Array<Float?>>>> {
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
