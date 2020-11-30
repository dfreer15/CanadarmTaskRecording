using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Assets.LSL4Unity.Scripts;

// ###################################################################
// This script is connected to the obstacles that float around the ISS
// ###################################################################


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

    // Initializes the start time
    void Start () {
        startTime = Time.time;
    }
	
    // Update is called once per frame
    void Update () {
    	
	// Checks for collisions with this object
	if (collided)
        {
            Color cp_color = collision_panel.color;
            cp_color.a -= 0.001f;
            collision_panel.color = cp_color;
        }

	// Moves objects if they are meant to be moving
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

    // Slows down obstacle until next update time is reached
    void Update_Pose()
    {
        transform.position += obj_vel;
        obj_vel.x *= 0.999f;
        obj_vel.y *= 0.999f;
        obj_vel.z *= 0.999f;
    }

    // Called if something collides with this object
    public void OnCollisionEnter(Collision collision)
    {
        string[] mark = new string[2];

        if (!collided)
        {
            // color change to make it clear when user collides
            Color cp_color = collision_panel.color;
            cp_color.a += 0.5f;
            collision_panel.color = cp_color;

	    // Logs time of collision and updates points
            c_control.event_log.Add(collision.gameObject.name + " collided with " + gameObject.name);
            c_control.event_log.Add(Time.time.ToString());
            point_counter.Update_Points(-100);
            collided = true;

            // Send type of collision to LSL for synchronization
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
            else{                                       // Collision with the ISS
                mark[0] = "-55";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);
            }
        }
    }
}
