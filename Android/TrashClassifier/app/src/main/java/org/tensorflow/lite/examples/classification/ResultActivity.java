package org.tensorflow.lite.examples.classification;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class ResultActivity extends AppCompatActivity {
    TextView resultTextView;
    ImageView resultImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        resultTextView = findViewById(R.id.resultTextView);
        resultImage = (ImageView) findViewById(R.id.imageView1);

        String recyclable = "recylable";
        Bundle bn = getIntent().getExtras();
        String name = bn.getString("name");
        String state = bn.getString("state");
        resultTextView.setText("This is " + String.valueOf(name) +".\nIt's " + String.valueOf(state));
        if (recyclable.equals(String.valueOf(state))) {
            resultImage.setImageResource(R.drawable.recyclable);
        } else{
            resultImage.setImageResource(R.drawable.unrecyclable);
        }




    }
}
