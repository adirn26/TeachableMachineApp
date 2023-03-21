package com.example.teachablemachinedemo;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.teachablemachinedemo.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.metadata.schema.ImageSize;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize=224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
//                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//                    startActivityForResult(cameraIntent, 1);
                    startActivityForResult(new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE), 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

            TensorImage tensorImage = new TensorImage(DataType.UINT8);
            tensorImage.load(image);
            ByteBuffer byteBuffer = tensorImage.getBuffer();
//            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize* imageSize*3);
//            byteBuffer.order(ByteOrder.nativeOrder());

//            int [] intValues = new int[imageSize*imageSize];
//            image.getPixels(intValues, 0, image.getWidth(), 0 , 0 , image.getWidth(), image.getHeight());
//            int pixel = 0;
//            for(int i=0;i<imageSize; i++){
//                for(int j=0;j<imageSize; j++){
//                    int val = intValues[pixel++];
//                    byteBuffer.putFloat(((val>>16)&0xFF)*(1.f/255.f));
//                    byteBuffer.putFloat(((val>>8)&0xFF)*(1.f / 255.f));
//                    byteBuffer.putFloat((val&0xff)*(1.f/255.f));
//                }
//            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            float[] confidences = outputFeature0.getFloatArray();
            int maxPos= 0;
            float  maxConfidence = 0;

            for(int i=0;i<confidences.length; i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"One", "Two"};

            result.setText(classes[maxPos]);

            String s="";
            for(int i=0;i<classes.length; i++){
                s+=String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            confidence.setText(s);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            // There are no request codes
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getHeight(), image.getWidth());
            image= ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }

    }
}