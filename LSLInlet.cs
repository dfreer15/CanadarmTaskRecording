using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LSL;

public class LSLInlet : MonoBehaviour
{
    public bool use_LSL_inlet = false;

    private liblsl.StreamInfo[] lslStreamInfo;
    private liblsl.StreamInlet lslInlet;
    public string lslStreamName = "NB-2015.10.76";
    public string lslStreamType = "EEG_data";           // Must find real stream type?

    private int lslChannelCount = 32;    // Could be 38? Depending on config file for EEG

    //Assuming that markers are never sent in regular intervals
    //private double nominal_srate = liblsl.IRREGULAR_RATE;

    // For EEG data, should be 250 Hz
    private double nominal_srate = 250;

    private const liblsl.channel_format_t lslChannelFormat = liblsl.channel_format_t.cf_float32;

    private float[] EEG_data;
    private float[,] data_EEG;
    private double[] timestamps;

    private List<double[]> timestamps_final;
  
    private void Awake()
    {
        if (use_LSL_inlet)
        {
            lslStreamInfo = liblsl.resolve_stream("name", lslStreamName);
            lslInlet = new liblsl.StreamInlet(lslStreamInfo[0]);
        }
    }

    // Use this for initialization
    void Start()
    {
        EEG_data = new float[lslChannelCount];
        data_EEG = new float[512, lslChannelCount];
        timestamps = new double[512];
    }

    // Update is called once per frame
    void Update()
    {
        float[] EEG_data_sample = get_sample();
        //print(EEG_data_sample[0] + " " + EEG_data_sample[1] + " " + EEG_data_sample[2] + " " + EEG_data_sample[3]);

        //float[,] EEG_data_samples = get_chunk();  // This is wayyyyyy too slow
        //print(EEG_data_samples[0,0]);
    }

    public float[] get_sample()
    {
        lslInlet.pull_sample(EEG_data, 0.1f);
        return EEG_data;
    }

    public float[,] get_chunk()
    {
        lslInlet.pull_chunk(data_EEG, timestamps);
        for (int i = 0; i < 512; i++)
        {
            for (int j = 0; j < lslChannelCount; j++)
            {
                print(data_EEG[i, j]);
            }
        }
        //timestamps_final.Add(timestamps);
        return data_EEG;
    }

    private void OnApplicationQuit()
    {
        //print(timestamps_final);
    }
}

