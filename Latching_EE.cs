using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Assets.LSL4Unity.Scripts;

public class Latching_EE : MonoBehaviour {

    public GameObject Hamlyn_Module;
    public CanadarmControl canadarm_control;
    public Camera columbus_camera;
    public Camera EE_camera;

    public LSLMarkerStream markerStream;

    public Points point_counter;

    bool three_panel = false;
    bool connected_to_rm = false;

    // Use this for initialization
    void Start() {

    }

    // Update is called once per frame
    void Update() {

    }

    private void OnTriggerStay(Collider other)
    {
        string[] mark = new string[2];
        if (other.gameObject.name == "Grapple_Fixture_Collider")
        {
            if (!connected_to_rm)
            {
                if (Input.GetKeyDown("joystick button 1"))  // If user presses the "x" button
                {
                    Hamlyn_Module.transform.parent = transform;
                    print("Hamlyn Module engaged");
                    print(Time.time);
                    print(System.DateTime.Now);

                    canadarm_control.event_log.Add("Hamlyn Module engaged");
                    canadarm_control.event_log.Add(Time.time.ToString());
                    canadarm_control.event_log.Add(System.DateTime.Now.ToString());

                    if (three_panel)
                    {
                        canadarm_control.cameras[3] = columbus_camera.gameObject;
                        columbus_camera.depth = 2;
                    }

                    Vector3 grap_fixt_pos = other.transform.position;
                    Quaternion grap_fixt_orient = other.transform.rotation;
                    Vector3 this_pos = transform.position;
                    Quaternion this_orient = transform.rotation;

                    float dist_between = Vector3.Distance(grap_fixt_pos, this_pos);
                    float angle_between = Quaternion.Angle(grap_fixt_orient, this_orient);

                    int points_change_dist = Mathf.RoundToInt(100.0f / dist_between);
                    int points_change_angle = Mathf.RoundToInt(5000.0f / angle_between);
                    int total_points_change = points_change_dist + points_change_angle;

                    print("Distance:");
                    print(dist_between);
                    print(points_change_dist);
                    print("Angle:");
                    print(angle_between);
                    print(points_change_angle);

                    canadarm_control.event_log.Add("Distance:");
                    canadarm_control.event_log.Add(dist_between.ToString("f3"));
                    canadarm_control.event_log.Add(points_change_dist.ToString());
                    canadarm_control.event_log.Add("Angle:");
                    canadarm_control.event_log.Add(angle_between.ToString("f2"));
                    canadarm_control.event_log.Add(points_change_angle.ToString());

                    point_counter.Update_Points(total_points_change);
                    // Send to LSL
                    mark[0] = "1000";
                    mark[1] = total_points_change.ToString();

                    markerStream.Write(mark);
                    //print("Columbus_attachpoint - xpoint pressed -- LSL sent");

                    //markerStream.Write(total_points_change.ToString());

                    connected_to_rm = true;
                }
            }
        }

        if (other.gameObject.name == "Columbus_attachpoint")
        {
            //string[] mark = new string[2];
            if (Input.GetKeyDown("joystick button 3"))   // If user presses the "triangle" button
            {
                Hamlyn_Module.transform.parent = null;
                print("Hamlyn Module unloaded");
                print(Time.time);
                print(System.DateTime.Now);

                canadarm_control.event_log.Add("Hamlyn Module unloaded");
                canadarm_control.event_log.Add(Time.time.ToString());
                canadarm_control.event_log.Add(System.DateTime.Now.ToString());

                if (three_panel)
                {
                    canadarm_control.cameras[3] = EE_camera.gameObject;
                    columbus_camera.depth = 0;
                }

                Vector3 grap_fixt_pos = other.transform.position;
                Quaternion grap_fixt_orient = other.transform.rotation;
                Vector3 this_pos = transform.position;
                Quaternion this_orient = transform.rotation;

                float dist_between = Vector3.Distance(grap_fixt_pos, this_pos);
                float angle_between = Quaternion.Angle(grap_fixt_orient, this_orient);

                int points_change_dist = Mathf.RoundToInt(100.0f / dist_between);
                int points_change_angle = Mathf.RoundToInt(5000.0f / angle_between);
                int total_points_change = points_change_dist + points_change_angle;

                //print("Distance:");
                //print(dist_between);
                //print(points_change_dist);
                //print("Angle:");
                //print(angle_between);
                //print(points_change_angle);

                canadarm_control.event_log.Add("Distance:");
                canadarm_control.event_log.Add(dist_between.ToString("f3"));
                canadarm_control.event_log.Add(points_change_dist.ToString());
                canadarm_control.event_log.Add("Angle:");
                canadarm_control.event_log.Add(angle_between.ToString("f2"));
                canadarm_control.event_log.Add(points_change_angle.ToString());

                point_counter.Update_Points(total_points_change);
                // Send to LSL
                //markerStream.Write(total_points_change.ToString(), Time.time);
                mark[0] = "2000";
                mark[1] = total_points_change.ToString();

                markerStream.Write(mark);
                //print("Columbus_attachpoint - triangle pressed-- LSL sent");

                canadarm_control.Save_Data();
            }
        }
    }
}
