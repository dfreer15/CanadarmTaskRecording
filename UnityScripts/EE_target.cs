using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EE_target : MonoBehaviour {

    CanadarmControl c_control;
    float EE_max_speed = 0.008f;
    Vector3 follow_offset;
    public Transform EE;
    public GameObject canadarm_base;

    // Initializes c_control object
    void Start () {
        c_control = GameObject.Find("Canadarm").GetComponent<CanadarmControl>();
    }
	
    // Moves the end-effector target
    public void move_ee_target(GameObject camera, float direction)
    {
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

	// Determines coordinate frame for traditional control
        if (c_control.ee_mode)
            camera = EE.gameObject;
        if (c_control.base_mode)
            camera = canadarm_base;

	// Determines direction of movement
        Quaternion cam_rot = camera.transform.rotation;
        Vector3 cam_forward = camera.transform.forward;

        Vector3 z_direc = cam_rot * unit_vector_z;
        Vector3 y_direc = cam_rot * unit_vector_y;
        Vector3 x_direc = cam_rot * unit_vector_x;
        Vector3 all_direc = new Vector3(x_direc[2], y_direc[2], z_direc[2]);

	// Moves EE target
        transform.position = transform.position + direction * z_direc *EE_max_speed;
    }

    public void move_ee_target_lr(GameObject camera, float direction)
    {
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

        if (c_control.ee_mode)
            camera = EE.gameObject;
        if (c_control.base_mode)
            camera = canadarm_base;

        Quaternion cam_rot = camera.transform.rotation;

        Vector3 z_direc = cam_rot * unit_vector_z;
        Vector3 y_direc = cam_rot * unit_vector_y;
        Vector3 x_direc = cam_rot * unit_vector_x;
        Vector3 all_direc = new Vector3(x_direc[1], y_direc[1], z_direc[1]);

        transform.position = transform.position + direction * x_direc *EE_max_speed;
    }

    public void move_ee_target_ud(GameObject camera, float direction)
    {
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

        if (c_control.ee_mode)
            camera = EE.gameObject;
        if (c_control.base_mode)
            camera = canadarm_base;

        //Quaternion cam_rot = camera.transform.localRotation;
        Quaternion cam_rot = camera.transform.rotation;

        Vector3 z_direc = cam_rot * unit_vector_z;
        Vector3 y_direc = cam_rot * unit_vector_y;
        Vector3 x_direc = cam_rot * unit_vector_x;
        Vector3 all_direc = new Vector3(x_direc[0], y_direc[0], z_direc[0]);

        //transform.position = transform.position + direction * all_direc / 700;
        transform.position = transform.position + direction * y_direc *EE_max_speed;
    }

    public void turn_ee_target_lr(GameObject camera, float direction)
    {
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

        //Quaternion cam_rot = camera.transform.localRotation;
        Quaternion cam_rot = camera.transform.rotation;

        Vector3 z_direc = cam_rot * unit_vector_z;
        Vector3 y_direc = cam_rot * unit_vector_y;
        Vector3 x_direc = cam_rot * unit_vector_x;

        transform.Rotate(y_direc* direction / 50);
    }

    public void turn_ee_target_ud(GameObject camera, float direction)
    {
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

        //Quaternion cam_rot = camera.transform.localRotation;
        Quaternion cam_rot = camera.transform.rotation;

        Vector3 z_direc = cam_rot * unit_vector_z;
        Vector3 y_direc = cam_rot * unit_vector_y;
        Vector3 x_direc = cam_rot * unit_vector_x;

        transform.Rotate(x_direc * direction / 50);
    }

    public void move_ee_target_joint()
    {
        Vector3 EE_pos = EE.position;
        Vector3 this_pos = transform.position;

        Vector3 offset = EE_pos - this_pos;

        int i = 1;
    }
}
