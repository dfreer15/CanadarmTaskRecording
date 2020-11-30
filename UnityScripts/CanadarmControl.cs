using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;
using System.IO;
using Assets.LSL4Unity.Scripts;

public class CanadarmControl : MonoBehaviour {

    public string user_name = "Your Name";
    public bool use_obstacles = false;
    public bool time_pressure = false;
    public float latency = 0f;

    //public bool three_panel = false;
    //public bool four_panel = true;
    public bool cam_control;
    public bool traditional_control;
    public bool ee_mode;
    public bool base_mode;
    public bool joint_mode;
    public Text mode_text;

    bool record_EEG = true;
    bool control_EE_orient = true;

    private List<Vector4> EE_move_input = new List<Vector4>();
    private List<float> timestamps_move = new List<float>();
    int move_EE_int = 0;
    private List<Vector3> EE_orient_input = new List<Vector3>();
    private List<float> timestamps_eeo = new List<float>();
    int eeo_int = 0;
    private List<Vector3> Cam_turn_input = new List<Vector3>();
    private List<float> timestamps_ct = new List<float>();
    int ct_int = 0;

    private List<float[]> robot_joint_pos_angle = new List<float[]>();
    private List<float> timestamps = new List<float>();

    public List<string> event_log = new List<string>();
    public bool log_start_info = false;
    public bool task_started;

    public RobotJoint[] all_joints;
    public Slider[] all_joints_ui;

    public GameObject Base_EE;
    public GameObject latching_EE;
    public GameObject latching_EE_target;

    public GameObject camera2outline;
    public GameObject camera3outline;
    public GameObject camera4outline;
    public GameObject camera11outline;
    public GameObject camera12outline;
    public GameObject camera21outline;
    public GameObject camera22outline;

    public GameObject[] cameras;
    public GameObject columbus_camera;
    public GameObject active_cam;

    public CanadarmKinematics kinematics;

    public Points points;

    float[] angles = new float[7];

    int cam_num = 0;
    int total_cams = 3;

    public bool target_pos_changed = false;
    bool gui_only = false;
    bool EE_orient_changed = false;
    
    Quaternion EE_coll_rotation;
    Quaternion initial_rotation;

    EE_target ee_target;

    public LSLMarkerStream markerStream;
    string[] mark = new string[2];

    string start_time;
    float startTime;

    
    void Start() {
        
        startTime = Time.time;

        ee_target = latching_EE_target.GetComponent<EE_target>();
        initial_rotation = all_joints[0].transform.rotation;
        
        // Get initial (1D) angles of each robotic joint
        angles[0] = all_joints[0].transform.localEulerAngles[2];
        angles[1] = all_joints[1].transform.localEulerAngles[2];
        angles[2] = all_joints[2].transform.localEulerAngles[1];
        angles[3] = all_joints[3].transform.localEulerAngles[1];
        angles[4] = all_joints[4].transform.localEulerAngles[1];
        angles[5] = all_joints[5].transform.localEulerAngles[2];
        angles[6] = all_joints[6].transform.localEulerAngles[2];

        // Cleans each angle to be between -180 and 180 degrees, then converts to radians
        for (int i = 0; i < angles.Length; i++)
        {
            if (angles[i] > 180)
                angles[i] -= 360;
            angles[i] = angles[i] * Mathf.PI / 180.0f;
        }

        // Initializes kinematics
        kinematics.initialize_FK();
        //kinematics.forward_kinematics(angles);  // Used for debugging
        kinematics.build_canadarm_jacobian(angles);

        // Initializes LSL stream - the python code will read this in to begin recording EEG data
        if (record_EEG)
        {
            mark[0] = "100";
            mark[1] = (Time.time).ToString();
            markerStream.Write(mark);
        }
    }

    // Update is called once per frame
    void Update() {

        gui_only = false;

        if (log_start_info)
        {
            // Initializes LSL stream - the python code will read this in to begin recording EEG data
            if (record_EEG)
            {
                mark[0] = "100";
                mark[1] = (Time.time).ToString();
                markerStream.Write(mark);
            }
            
            // Logs start time
            start_time = System.DateTime.Now.ToString("s");
            start_time = start_time.Replace(':', '-');
            print("Start Time: ");
            print(start_time);
            event_log.Add("Start Time: ");
            event_log.Add(start_time);

            log_start_info = false;
            task_started = true;
        }
        
        if (task_started)
        {
            // Receives input from joystick and updates selected camera outlines
            getJoystickData();
            Update_Camera_Outline();
            
            latching_EE_target.transform.parent = null;

            // Get current (1D) angles of each robotic joint
            angles[0] = all_joints[0].transform.localEulerAngles[2];
            angles[1] = all_joints[1].transform.localEulerAngles[2];
            angles[2] = all_joints[2].transform.localEulerAngles[1];
            angles[3] = all_joints[3].transform.localEulerAngles[1];
            angles[4] = all_joints[4].transform.localEulerAngles[1];
            angles[5] = all_joints[5].transform.localEulerAngles[2];
            angles[6] = all_joints[6].transform.localEulerAngles[2];

            // Cleans each angle to be between -180 and 180 degrees, then converts to radians
            for (int i = 0; i < angles.Length; i++)
            {
                if (angles[i] > 180)
                    angles[i] -= 360;
                angles[i] = angles[i] * Mathf.PI / 180.0f;
            }

            // Either updates joint angles via inverse kinematics, or updates target position via forward kinematics
            if (target_pos_changed)
                Inverse_Kinematics(latching_EE_target, angles);   // angles in radians
            else
                ee_target.transform.position = kinematics.forward_kinematics(angles);

            // Includes consideration for latency
            if (latency > 0.0f)
            {
                // If enough lag has occurred, move the robotic components as appropriate (left joystick and L2/R2)
                if (Time.time > timestamps_move[move_EE_int] + latency)
                {
                    Vector4 move_command = EE_move_input[move_EE_int];
  
                    // -2f indicates that the robot end-effector is moving in Cartesian space
                    if (move_command[3] == -2f)
                    {
                        ee_target.move_ee_target(active_cam, move_command[0]);
                        ee_target.move_ee_target_lr(active_cam, move_command[1]);
                        ee_target.move_ee_target_ud(active_cam, move_command[2]);
                    }
                    else  // The robot's proximal joints are directly moved in joint space (traditional control)
                    {
                        all_joints[0].joint_target += move_command[0] / 30.0f;
                        all_joints[1].joint_target += move_command[1] / 30.0f;
                        all_joints[2].joint_target += move_command[2] / 30.0f;
                        all_joints[3].joint_target += move_command[3] / 30.0f;
                    }

                    move_EE_int += 1;  // Iterates through the saved controller input
                }

                // Control for camera turn input (d-pad)
                if (Time.time > timestamps_ct[ct_int] + latency)
                {
                    // Turns currently selected camera
                    active_cam.transform.localEulerAngles = active_cam.transform.localEulerAngles + Cam_turn_input[ct_int]/10.0f;

                    ct_int += 1;  // Iterates through saved controller input
                }

                // End-effector joint control (final three DoF, right joystick and circle/square)
                if (Time.time > timestamps_eeo[eeo_int] + latency)
                {
                    // Turn the last three joints of the Canadarm2
                    Vector3 turn_command = EE_orient_input[eeo_int];

                    //float cur_joint_5 = all_joints[5].joint_target;
                    all_joints[5].joint_target += turn_command[0]/30.0f;

                    //float cur_joint_4 = all_joints[4].joint_target;
                    all_joints[4].joint_target += turn_command[1]/30.0f;

                    //float cur_joint_6 = all_joints[6].joint_target;
                    all_joints[6].joint_target += turn_command[2]/30.0f;

                    eeo_int += 1;  // Iterates through saved controller input
                }

                if((Time.time > timestamps_eeo[eeo_int] + latency) && (Time.time > timestamps_move[move_EE_int] + latency) && (Time.time > timestamps_ct[ct_int] + latency))
                    target_pos_changed = false;
            }
            else
                target_pos_changed = false;

            // Updates all robot joint values
            for (int i = 0; i < all_joints.Length - 1; i++)
            {
                all_joints[i].UpdateJoint();
            }

            record_joints_data();
            
        }

    }

    // Gets data from the Playstation controller
    private void getJoystickData()
    {
        Get_Cam_JS();

        if (control_EE_orient)
            Get_EE_Orient_JS();
        if (joint_mode)
            Get_Joint_Rots_JS();
        else
            Get_EE_Pos_JS();
    }

    // Gets data from parts of controller that control the cameras
    private void Get_Cam_JS()
    {
        Vector3 cam_turn = new Vector3(0.0f, 0.0f, 0.0f);
        bool cam_turn_changed = false;

        // If R1 is pressed
        if (Input.GetKeyDown("joystick button 5"))
        {
            // If R1 and L1 are pressed at the same time, and traditional control mode is selected
            if (traditional_control && Input.GetKey("joystick button 4"))
                switch_modes();

            // Switch cameras
            cam_num = cam_num + 1;
            if (cam_num > total_cams)
                cam_num = 0;
 
            active_cam = cameras[cam_num];
            target_pos_changed = true;

            
        }
        else if (Input.GetKeyDown("joystick button 4"))    // If L1 is pressed
        {
            // If R1 and L1 are pressed at the same time, and traditional control mode is selected
            if (traditional_control && Input.GetKey("joystick button 5"))
                switch_modes();

            // Switch cameras
            cam_num = cam_num - 1;
            if (cam_num < 0)
                cam_num = total_cams;

            active_cam = cameras[cam_num];
            target_pos_changed = true;
        }

        // Checks d-pad left or right buttons
        if (Math.Abs(Input.GetAxis("DpadLR")) > 0.1)
        {
            // Saves value to turn camera
            cam_turn[1] = -Input.GetAxis("DpadLR");
            cam_turn_changed = true;
        }

        // Checks d-pad up or down buttons
        if (Math.Abs(Input.GetAxis("DpadUD")) > 0.1)
        {
            // Saves value to turn camera
            cam_turn[0] = Input.GetAxis("DpadUD");
            cam_turn_changed = true;
        }

        // Turns camera automatically if no latency is present
        if (latency == 0.0f)
        {
            active_cam.transform.localEulerAngles = active_cam.transform.localEulerAngles + cam_turn / 10.0f; 
        }

        // Saves input and tells camera to turn
        cam_turn_changed = true;
        if (cam_turn_changed)
        {
            target_pos_changed = true;
            Cam_turn_input.Add(cam_turn);
            timestamps_ct.Add(Time.time);
        }
    }

    // Gets input from controller parts that control the end-effector orientation (right joystick and circle/square)
    private void Get_EE_Orient_JS()
    {
        Vector3 EE_orient = new Vector3(0.00f, 0.00f, 0.00f);
        EE_orient_changed = false;
        if (Math.Abs(Input.GetAxis("JoystickTurnEELeftRight")) > 0.1)
        {
            // Control Yaw of EE
            EE_orient_changed = true;
            float cur_joint = all_joints[5].joint_target;
            EE_orient[0] = Input.GetAxis("JoystickTurnEELeftRight");
            
            // Directly turns joint if latency is zero
            if (latency == 0.0f)
                all_joints[5].joint_target = cur_joint + EE_orient[0]/30.0f;
        }
        
        if (Math.Abs(Input.GetAxis("JoystickTurnEEUpDown")) > 0.1)
        {
            // Control Pitch of EE
            EE_orient_changed = true;
            float cur_joint = all_joints[4].joint_target;
            EE_orient[1] = Input.GetAxis("JoystickTurnEEUpDown");
            
            // Directly turns joint if latency is zero
            if (latency == 0.0f)
                all_joints[4].joint_target = cur_joint + EE_orient[1]/30.0f;
        }
        if (Input.GetKey("joystick button 0"))   // Square button
        {
            // Roll EE counterclockwise
            EE_orient_changed = true;
            float cur_joint = all_joints[6].joint_target;
            EE_orient[2] = -1.0f;
            
            // Directly turns joint if latency is zero
            if (latency == 0.0f)
                all_joints[6].joint_target = cur_joint + EE_orient[2]/30.0f;
        }
        else if (Input.GetKey("joystick button 2"))    // Circle button
        {
            // Roll EE clockwise
            EE_orient_changed = true;
            float cur_joint = all_joints[6].joint_target;
            EE_orient[2] = 1.0f;
            
            // Directly turns joint if latency is zero
            if (latency == 0.0f)
                all_joints[6].joint_target = cur_joint + EE_orient[2]/30.0f;
        }

        // Saves input to eventually tell EE to turn (if latency > 0)
        EE_orient_changed = true;
        if (EE_orient_changed)
        {
            target_pos_changed = true;
            EE_orient_input.Add(EE_orient);
            timestamps_eeo.Add(Time.time);
        }
    }

    // Get joystick components controlling end-effector position
    private void Get_EE_Pos_JS()
    {
        Vector4 EE_move = new Vector4(0, 0, 0, -2);
        bool EE_pos_changed = false;

        // L2 button
        if (Input.GetKey("joystick button 7"))
        {
            target_pos_changed = true;
            EE_pos_changed = true;
            EE_move[0] = -1.0f;
            if (latency == 0.0f)
                ee_target.move_ee_target(active_cam, -1.0f);
        }

        // R2 button
        if (Input.GetKey("joystick button 6"))
        {
            target_pos_changed = true;
            EE_pos_changed = true;
            EE_move[0] = 1.0f;
            if (latency == 0.0f)
                ee_target.move_ee_target(active_cam, 1.0f);
        }

        // Left joystick (left/right movement)
        if (Math.Abs(Input.GetAxis("Horizontal")) > 0.1)
        {
            target_pos_changed = true;
            EE_pos_changed = true;
            EE_move[1] = Input.GetAxis("Horizontal");
            if (latency == 0.0f)
                ee_target.move_ee_target_lr(active_cam, EE_move[1]);
        }

        // Left joystick (up/down movement)
        if (Math.Abs(Input.GetAxis("Vertical")) > 0.1)
        {
            target_pos_changed = true;
            EE_pos_changed = true;
            EE_move[2] = Input.GetAxis("Vertical");
            if (latency == 0.0f)
                ee_target.move_ee_target_ud(active_cam, EE_move[2]);
        }

        // Saves joystick commands for log and/or later movement
        EE_pos_changed = true;
        if (EE_pos_changed)
        {
            EE_move_input.Add(EE_move);
            timestamps_move.Add(Time.time);
        }
    }

    // Sets outlines for camera views to be true or false depending on which is selected
    private void Update_Camera_Outline()
    {
        if (cam_num == 1)
        {
            camera2outline.SetActive(true);
            camera3outline.SetActive(false);
            camera4outline.SetActive(false);

            camera11outline.SetActive(true);
            camera12outline.SetActive(false);
            camera21outline.SetActive(false);
            camera22outline.SetActive(false);
        }
        else if (cam_num == 2)
        {
            camera2outline.SetActive(false);
            camera3outline.SetActive(true);
            camera4outline.SetActive(false);

            camera11outline.SetActive(false);
            camera12outline.SetActive(true);
            camera21outline.SetActive(false);
            camera22outline.SetActive(false);
        }
        else if (cam_num == 3)
        {
            camera2outline.SetActive(false);
            camera3outline.SetActive(false);
            camera4outline.SetActive(true);

            camera11outline.SetActive(false);
            camera12outline.SetActive(false);
            camera21outline.SetActive(true);
            camera22outline.SetActive(false);
        }
        else
        {
            camera2outline.SetActive(false);
            camera3outline.SetActive(false);
            camera4outline.SetActive(false);

            camera11outline.SetActive(false);
            camera12outline.SetActive(false);
            camera21outline.SetActive(false);
            camera22outline.SetActive(true);
        }
    }

    // Gets data from pats of joystick that control proximal joints - if traditional joint mode is selected
    void Get_Joint_Rots_JS()
    {
        Vector4 EE_orient = new Vector4(0, 0, 0, 0);
        target_pos_changed = false;

        // R2 button
        if (Input.GetKey("joystick button 7"))
        {
            EE_orient_changed = true;
            float cur_joint = all_joints[0].joint_target;
            EE_orient[0] = -1.0f;
            if (latency == 0.0f)
                all_joints[0].joint_target = cur_joint + EE_orient[0] / 30.0f;
        }

        // L2 button
        if (Input.GetKey("joystick button 6"))
        {
            float cur_joint = all_joints[0].joint_target;
            EE_orient_changed = true;
            EE_orient[0] = 1.0f;
            if (latency == 0.0f)
                all_joints[0].joint_target = cur_joint + EE_orient[0] / 30.0f;
        }

        // Left joystick - left/right
        if (Math.Abs(Input.GetAxis("Horizontal")) > 0.1)
        {
            float cur_joint = all_joints[1].joint_target;
            EE_orient_changed = true;
            EE_orient[1] = Input.GetAxis("Horizontal");
            if (latency == 0.0f)
                all_joints[1].joint_target = cur_joint + EE_orient[1] / 30.0f;
        }

        // Left joystick - up/down
        if (Math.Abs(Input.GetAxis("Vertical")) > 0.1)
        {
            float cur_joint = all_joints[2].joint_target;
            EE_orient_changed = true;
            EE_orient[2] = Input.GetAxis("Vertical");
            if (latency == 0.0f)
                all_joints[2].joint_target = cur_joint + EE_orient[2] / 30.0f;
        }

        if (Input.GetKey("joystick button 1"))    // x is pressed
        {
            float cur_joint = all_joints[3].joint_target;
            EE_orient_changed = true;
            EE_orient[3] = 1.0f;
            if (latency == 0.0f)
                all_joints[3].joint_target = cur_joint + EE_orient[3] / 30.0f;
        }

        if (Input.GetKey("joystick button 3"))   // triangle is pressed
        {
            float cur_joint = all_joints[3].joint_target;
            EE_orient_changed = true;
            EE_orient[3] = -1.0f;
            if (latency == 0.0f)
                all_joints[3].joint_target = cur_joint + EE_orient[3] / 30.0f;
        }

        // Adds controller input to array 
        EE_orient_changed = true;
        if (EE_orient_changed)
        {
            EE_move_input.Add(EE_orient);
            timestamps_move.Add(Time.time);
        }
    }

    // Rotates through different (traditional) control modes when R1 and L1 are pressed
    void switch_modes()
    {
        if (ee_mode)
        {
            base_mode = true;
            joint_mode = false;
            ee_mode = false;
            mode_text.text = "BASE MODE";
        }
        else if (base_mode)
        {
            joint_mode = true;
            ee_mode = false;
            base_mode = false;
            mode_text.text = "JOINT MODE";
        }
        else if (joint_mode)
        {
            ee_mode = true;
            base_mode = false;
            joint_mode = false;
            mode_text.text = "EE MODE";
        }
    }

    // Structures all robot joint data to be saved to a file at the end of the task
    private void record_joints_data()
    {
        float[] joints_pos_angles = new float[31];
        joints_pos_angles[0] = all_joints[0].transform.position.x;
        joints_pos_angles[1] = all_joints[0].transform.position.y;
        joints_pos_angles[2] = all_joints[0].transform.position.z;
        joints_pos_angles[3] = all_joints[1].transform.position.x;
        joints_pos_angles[4] = all_joints[1].transform.position.y;
        joints_pos_angles[5] = all_joints[1].transform.position.z;
        joints_pos_angles[6] = all_joints[2].transform.position.x;
        joints_pos_angles[7] = all_joints[2].transform.position.y;
        joints_pos_angles[8] = all_joints[2].transform.position.z;
        joints_pos_angles[9] = all_joints[3].transform.position.x;
        joints_pos_angles[10] = all_joints[3].transform.position.y;
        joints_pos_angles[11] = all_joints[3].transform.position.z;
        joints_pos_angles[12] = all_joints[4].transform.position.x;
        joints_pos_angles[13] = all_joints[4].transform.position.y;
        joints_pos_angles[14] = all_joints[4].transform.position.z;
        joints_pos_angles[15] = all_joints[5].transform.position.x;
        joints_pos_angles[16] = all_joints[5].transform.position.y;
        joints_pos_angles[17] = all_joints[5].transform.position.z;
        joints_pos_angles[18] = all_joints[6].transform.position.x;
        joints_pos_angles[19] = all_joints[6].transform.position.y;
        joints_pos_angles[20] = all_joints[6].transform.position.z;
        joints_pos_angles[21] = all_joints[7].transform.position.x;
        joints_pos_angles[22] = all_joints[7].transform.position.y;
        joints_pos_angles[23] = all_joints[7].transform.position.z;
        joints_pos_angles[24] = all_joints[0].transform.localEulerAngles[2];
        joints_pos_angles[25] = all_joints[1].transform.localEulerAngles[2];
        joints_pos_angles[26] = all_joints[2].transform.localEulerAngles[1];
        joints_pos_angles[27] = all_joints[3].transform.localEulerAngles[1];
        joints_pos_angles[28] = all_joints[4].transform.localEulerAngles[1];
        joints_pos_angles[29] = all_joints[5].transform.localEulerAngles[2];
        joints_pos_angles[30] = all_joints[6].transform.localEulerAngles[2];

        robot_joint_pos_angle.Add(joints_pos_angles);
        timestamps.Add(Time.time);
    }
    
    
    
    
    
    // DO NOT DELETE --  THESE ARE USED FOR JOINT CONTROL
    public void Update_Joint_1_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[0].joint_target = new_ja;
        }
    }
    
    public void Update_Joint_2_target(float new_ja)
    {

        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[1].joint_target = new_ja;
        }
    }
    public void Update_Joint_3_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[2].joint_target = new_ja;
        }
    }
    public void Update_Joint_4_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[3].joint_target = new_ja;
        }
    }
    public void Update_Joint_5_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[4].joint_target = new_ja;
        }
    }

    public void Update_Joint_6_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[5].joint_target = new_ja;
        }
    }

    public void Update_Joint_7_target(float new_ja)
    {
        if (!gui_only)
        {
            target_pos_changed = false;
            latching_EE_target.transform.parent = latching_EE.transform;
            all_joints[6].joint_target = new_ja;
        }
    }


    
    // Inverse Kinematics
    public void Inverse_Kinematics(GameObject target, float[] angles_IK)
    {
        Vector3 base_pos = Base_EE.transform.position;
        float LearningRate = 0.002f;
        float DistanceThreshold = 0.2f;

        //float total_distance = DistanceFromTarget(target.transform.position, angles_IK) + rotational_distance(target, angles_IK);

        // If the end-effector is close to the target, inverse kinematics is not performed
        if (DistanceFromTarget(target.transform.position, angles_IK) < DistanceThreshold)
            return;

        // Starting with the last joint, compute partial gradients and perform gradient descent to find inverse kinematic solution
        for (int i = all_joints.Length - 2; i >=0;  i--)
        {
            float gradient = PartialGradient(target, i, angles_IK);
            angles_IK[i] -= LearningRate * gradient;
            all_joints[i].joint_target = angles_IK[i] * 180.0f/Mathf.PI;

            //total_distance = DistanceFromTarget(target.transform.position, angles_IK) + rotational_distance(target, angles_IK);

            if (DistanceFromTarget(target.transform.position, angles_IK) < DistanceThreshold)
                return;
        }
    }

    // Computes partial gradient
    public float PartialGradient(GameObject target, int i, float[] angles_PG)
    {
        float SamplingDistance = 0.01f;

        float angle = angles_PG[i];

        float f_x = DistanceFromTarget(target.transform.position, angles_PG);
        //f_x += rotational_distance(target, angles_PG);

        angles_PG[i] += SamplingDistance;

        float f_x_d = DistanceFromTarget(target.transform.position, angles_PG);
        //f_x_d += rotational_distance(target, angles_PG);
        float gradient = (f_x_d - f_x) / SamplingDistance;

        angles_PG[i] = angle;

        return gradient;

    }

    // Calculates distance from the target postiion
    public float DistanceFromTarget(Vector3 target, float[] angles_DfT)
    {
        //Vector3 point = ForwardKinematics(angles_DfT);

        Vector3 point = kinematics.forward_kinematics(angles_DfT);
        Vector3 true_pos = latching_EE.transform.position;
        //print(Vector3.Distance(point, target) + "    " + Vector3.Distance(latching_EE.transform.position, target));

        return Vector3.Distance(point, target);
    }

    // Saves data to files once the task is complete
    public void Save_Data()
    {
        points.GetFinalScore();

        // Sends a signal to LSL to stop python code
        mark[0] = "1000";
        mark[1] = (Time.time).ToString();
        markerStream.Write(mark);

        string filepath = "D:\\SpaceTrialRecordings2019_nirsEEG\\" + user_name;
        if (!Directory.Exists(filepath))
        {
            Directory.CreateDirectory(filepath);
        }
        filepath = filepath + "\\UnityData\\";
        if (!Directory.Exists(filepath))
        {
            Directory.CreateDirectory(filepath);
        }
        filepath = filepath + start_time;
        if (time_pressure)
            filepath += "_tp";
        if (use_obstacles)
            filepath += "_obs";
        if (latency > 0.0f)
        {
            filepath += "_lat";
            filepath += latency.ToString("f2").Replace('.','-');
        }

        string filepath_eem = filepath + "_EE_move_input.csv";
        StreamWriter writer_eem = new StreamWriter(filepath_eem);

        for (int i = 0; i < timestamps_move.Count; i++)
        {
            Vector3 ee_move = EE_move_input[i];
            writer_eem.WriteLine(timestamps_move[i] + "," + ee_move[0] + "," + ee_move[1] + "," + ee_move[2]);
        }

        string filepath_eeo = filepath + "_EE_orient_input.csv";
        StreamWriter writer_eeo = new StreamWriter(filepath_eeo);

        for (int i = 0; i < timestamps_eeo.Count; i++)
        {
            Vector3 ee_o = EE_orient_input[i];
            writer_eeo.WriteLine(timestamps_eeo[i] + "," + ee_o[0] + "," + ee_o[1] + "," + ee_o[2]);
        }

        string filepath_ct = filepath + "_CT_input.csv";
        StreamWriter writer_ct = new StreamWriter(filepath_ct);

        for (int i = 0; i < timestamps_ct.Count; i++)
        {
            Vector3 cam_turn = Cam_turn_input[i];
            writer_ct.WriteLine(timestamps_ct[i] + "," + cam_turn[0] + "," + cam_turn[1] + "," + cam_turn[2]);
        }


        string filepath_jpa = filepath + "_JPA.csv";
        StreamWriter writer_jpa = new StreamWriter(filepath_jpa);

        for (int i = 0; i < timestamps.Count; i++)
        {
            float[] joint_pos_angle = robot_joint_pos_angle[i];
            string line_to_write = timestamps[i] + ",";
            for (int j = 0; j < 31; j++)
            {
                line_to_write += joint_pos_angle[j];
                line_to_write += ",";
            }
            writer_jpa.WriteLine(line_to_write);
        }

        var writer_log = File.CreateText(filepath + "_log.txt");
        //StreamWriter writer_log = new StreamWriter(filepath_log);

        for (int i = 0; i < event_log.Count; i++)
        {
            print(event_log[i]);
            writer_log.WriteLine(i + "," + event_log[i]);
        }
        writer_log.Close();

        Application.Quit();     // Works on build > run
        //UnityEditor.EditorApplication.isPlaying = false;
        
    }
}
