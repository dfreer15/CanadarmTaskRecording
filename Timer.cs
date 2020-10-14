using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Timer : MonoBehaviour {

    public Text timerText;
    public Text timerText2;
    public float startTime;

    public float timer_threshold = 1000.0f;
    private float point_decrement = 5.0f;
    public Points point_counter;
    public CanadarmControl c_control;


	// Use this for initialization
	void Start () {
        startTime = Time.time;
        point_counter = GetComponent<Points>();
	}
	
	// Update is called once per frame
	void Update () {
        float t = Time.time - startTime;
        float t_rem = timer_threshold - t;

        if (c_control.time_pressure)
        {
            string minutes = ((int)t_rem / 60).ToString();
            string seconds = (t_rem % 60).ToString("f1");

            timerText.text = minutes + ":" + seconds;
            timerText2.text = minutes + ":" + seconds;

            // Turns timer yellow
            if (t_rem < timer_threshold * 0.3f)
                timerText.color = Color.yellow;

            // Turns timer red
            if (t_rem < timer_threshold * 0.1f)
                timerText.color = Color.red;

            // Ends task if time limit is reached
            if (t_rem < 0 && c_control.time_pressure)
                c_control.Save_Data();
        }       
        
        if (c_control.task_started)
        {
            if (t > point_decrement)
            {
                point_counter.Update_Points(-5);
                point_decrement = point_decrement + 5;
            }
        }
	}
}
