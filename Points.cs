using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Points : MonoBehaviour {

    public Text Score1;
    public Text Score2;
    public CanadarmControl c_control;
    public int score = 0;
    
    // Use this for initialization
	void Start () {
        //Update_Points(300);
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    public void Update_Points(int value_change)
    {
        score = score + value_change;

        Score1.text = score.ToString();
        Score2.text = score.ToString();
    }

    public void GetFinalScore()
    {
        print("Final Score: ");
        print(score);
        c_control.event_log.Add("Final Score: ");
        c_control.event_log.Add(score.ToString());
    }
}
