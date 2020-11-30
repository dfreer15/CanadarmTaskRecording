using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class C2_UI_Control : MonoBehaviour {

    // Game Elements
    public CanadarmControl c_control;
    public Timer timer;
    public GameObject timer_txt;
    public Light dir_light;

    // UI Elements
    public GameObject task_canvas;
    public GameObject ui_cam;
    public InputField user_name;
    public Slider latency_slider;
    public InputField latency_input;
    public GameObject all_obstacles;
    public Toggle obstacles_present;
    public Toggle obstacles_moving;
    public Toggle tp_toggle;
    public Slider tp_slider;
    public InputField tp_input;
    public Slider light_slider;
    public InputField light_input;
    public Toggle tc_toggle;
    public Toggle cc_toggle;

    public Points points;

    // When "Start Task" Button is pressed, set current values of each factor, log important information, and begin task
    public void Start_Task()
    {
        task_canvas.SetActive(true);
        ui_cam.SetActive(false);
        gameObject.SetActive(false);

	// Set values for user name, latency, obstacle presence, time pressure, and control mode
        c_control.user_name = user_name.text;
        c_control.latency = latency_slider.value;
        c_control.use_obstacles = obstacles_present.isOn;
        c_control.time_pressure = tp_toggle.isOn;
	
	// Determine whether to show timer during task (if time pressure is included)
        if (tp_toggle.isOn)  
            timer_txt.SetActive(true);
        else
            timer_txt.SetActive(false);
        timer.timer_threshold = tp_slider.value;
        
	// Determine control mode and whether to show control mode text during task
	c_control.traditional_control = tc_toggle.isOn;
        if (tc_toggle.isOn)
        {
            c_control.mode_text.text = "EE MODE";
            c_control.ee_mode = true;
        }
        else
        {
            c_control.mode_text.gameObject.SetActive(false);
            c_control.ee_mode = false;
        }

	// Set starting parameters and add light direction to log file string
        timer.startTime = Time.time;
        c_control.log_start_info = true;
        print("Light Direction: ");
        print(dir_light.transform.eulerAngles.ToString());
        c_control.event_log.Add("Light Direction: ");
        c_control.event_log.Add(dir_light.transform.eulerAngles.ToString());

	// Give user 300 points to start task
        points.Update_Points(300);
    }

    // Updates the latency slider if latency text has changed
    public void Update_Latency_Slider()
    {
        float result;
        if (float.TryParse(latency_input.text, out result))
        {
            float new_slider_val = float.Parse(latency_input.text);
            if (new_slider_val > 5)
                new_slider_val = 5;
            if (new_slider_val < 0)
                new_slider_val = 0;
            latency_slider.value = new_slider_val;
            latency_input.text = new_slider_val.ToString();
        }
        else
        {
            latency_slider.value = 0;
            latency_input.text = 0.ToString();
        }
    }

    // Updates latency text input if slider value has changed
    public void Update_Latency_Input()
    {
        latency_input.text = latency_slider.value.ToString();
    }

    // Changes "obstacles" group depending on user input, and activates obstacles if true
    public void Obstacles_Present_Changed()
    {
        if (obstacles_present.isOn)
        {
            obstacles_moving.interactable = true;
            all_obstacles.gameObject.SetActive(true);
        }
        else
        {
            obstacles_moving.isOn = false;
            obstacles_moving.interactable = false;
            all_obstacles.gameObject.SetActive(false);
        }
    }

    // Makes active obstacles begin to move randomly
    public void Obstacles_Moving_Changed()
    {
        if (obstacles_moving.isOn) 
	{
            foreach (Transform obstacle in all_obstacles.transform)
                obstacle.GetComponentInChildren<Collision_Points>().moving = true;
	}        
        else 
	{	
            foreach (Transform obstacle in all_obstacles.transform)
                obstacle.GetComponentInChildren<Collision_Points>().moving = false;
	}

    }

    // Updates "time pressure" UI block
    public void Add_Time_Pressure_Changed()
    {
        if (tp_toggle.isOn)
        {
            tp_slider.interactable = true;
            tp_input.interactable = true;
            tp_slider.value = 240;
        }
        else
        {
            tp_slider.interactable = false;
            tp_input.interactable = false;
        }
    }

    // Updates time pressure text input if the slider value has changed
    public void Update_TP_Input()
    {
        tp_input.text = tp_slider.value.ToString();
    }

    // Updates time pressure slider if text input value has changed
    public void Update_TP_Slider()
    {
        int result;
        if (int.TryParse(tp_input.text, out result))
        {
            int new_slider_val = int.Parse(tp_input.text);
            if (new_slider_val > 600)
                new_slider_val = 600;
            if (new_slider_val < 0)
                new_slider_val = 0;
            tp_slider.value = new_slider_val;
            tp_input.text = new_slider_val.ToString();
        }
        else
        {
            tp_slider.value = 0;
            tp_input.text = 0.ToString();
        }
    }

    // Reads in slider data to update light intensity, and updates lighting text input
    public void Update_Light_Input()
    {
        light_input.text = light_slider.value.ToString();
        Update_Light_Intensity();
    }

    // Reads in text input data to update light intensity, and updates lighting slider
    public void Update_Light_Slider()
    {
        float result;
        if (float.TryParse(light_input.text, out result))
        {
            float new_slider_val = float.Parse(light_input.text);
            if (new_slider_val > 1)
                new_slider_val = 1;
            if (new_slider_val < 0)
                new_slider_val = 0;
            light_slider.value = new_slider_val;
            light_input.text = new_slider_val.ToString();
        }
        else
        {
            light_slider.value = 0;
            light_input.text = 0.ToString();
        }
        Update_Light_Intensity();
    }

    // Updates light intensity
    public void Update_Light_Intensity()
    {
        dir_light.intensity = light_slider.value;
    }

    // Sets a random light direction
    public void Randomize_Light_Direction()
    {
        dir_light.transform.rotation = Random.rotation;
    }
}
