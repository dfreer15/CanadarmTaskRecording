using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

using Assets.LSL4Unity.Scripts;

public class Collision_Points : MonoBehaviour {

    public Points point_counter;
    public Image collision_panel;
    public float update_vel_time;
    public LSLMarkerStream markerStream;
    public bool moving = true;

    bool collided = false;
    private float startTime;
    float random_x;
    float random_y;
    float random_z;

    Vector3 obj_vel;

    public CanadarmControl c_control;

    // Use this for initialization
    void Start () {
        startTime = Time.time;
	}
	
	// Update is called once per frame
	void Update () {
        if (collided)
        {
            Color cp_color = collision_panel.color;
            cp_color.a -= 0.001f;
            collision_panel.color = cp_color;
        }

        if (moving)
        {
            if (Time.time - startTime > update_vel_time)
            {
                random_x = Random.Range(-1.0f, 1.0f) / 500.0f;
                random_y = Random.Range(-1.0f, 1.0f) / 500.0f;
                random_z = Random.Range(-1.0f, 1.0f) / 500.0f;

                obj_vel += new Vector3(random_x, random_y, random_z);

                startTime = Time.time;
            }
            Update_Pose();
        }
	}

    void Update_Pose()
    {
        /*Vector3 obj_pos = transform.position;

        obj_pos[0] += random_x/100.0f;
        obj_pos[1] += random_y/100.0f;
        obj_pos[2] += random_z/100.0f;*/

        //transform.position = obj_pos;
        transform.position += obj_vel;
        obj_vel.x *= 0.999f;
        obj_vel.y *= 0.999f;
        obj_vel.z *= 0.999f;
    }

    public void OnCollisionEnter(Collision collision)
    {
        string[] mark = new string[2];

        if (!collided)
        {
            // color change to make it clear when user collides
            Color cp_color = collision_panel.color;
            cp_color.a += 0.5f;
            collision_panel.color = cp_color;

            print(collision.gameObject.name + " collided with " + gameObject.name);
            print(Time.time);
            c_control.event_log.Add(collision.gameObject.name + " collided with " + gameObject.name);
            c_control.event_log.Add(Time.time.ToString());
            point_counter.Update_Points(-100);
            collided = true;

            // Send to LSL
            if (gameObject.name == "A1_Collider"){
                mark[0] = "-51";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);
            }
            else if (gameObject.name == "A2_Collider")
            {
                mark[0] = "-52";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);

            }
            else if (gameObject.name == "A3_Collider")
            {
                mark[0] = "-53";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);
            }
            else{       
                mark[0] = "-55";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);
            }

        }
        
    }
}
