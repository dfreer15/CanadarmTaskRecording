using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RobotJoint : MonoBehaviour {

    public Vector3 Axis;
    public Vector3 StartOffset;
    public GameObject PreviousJoint;

    public float joint_target = 0.0f;
    float speed = 0.1f;
    int rot_mask = 2;

    // Use this for initialization
    void Start () {
        //StartOffset = transform.position - PreviousJoint.transform.position;
        StartOffset = transform.localPosition;

        if (gameObject.name == "Joint3" || gameObject.name == "Joint4" || gameObject.name == "Joint5")
        {
            rot_mask = 1;
        }
    }
	
	// Update is called once per frame
	void Update () {
		
	}

    public void UpdateJoint()
    {
        Vector3 cur_angles = transform.localEulerAngles;
        float cur_ang;

        cur_ang = transform.localEulerAngles[rot_mask];

        if (cur_ang > 180)
        {
            cur_ang = cur_ang - 360;
        }

        if (joint_target > 180)
        {
            joint_target = joint_target - 360;
        }

        if(gameObject.name == "Joint6")
        {
            //print(cur_ang + "    " + joint_target);
        }

        //if (cur_ang < joint_target - speed)
        if (cur_ang < joint_target)
        {
            //print(cur_ang);
            Vector3 new_angles = cur_angles;
            new_angles[rot_mask] = new_angles[rot_mask] + speed;
            transform.localEulerAngles = new_angles;
        }
        //else if (cur_ang > joint_target + speed)
        else if (cur_ang > joint_target)
        {
            Vector3 new_angles = cur_angles;
            new_angles[rot_mask] = new_angles[rot_mask] - speed;
            transform.localEulerAngles = new_angles;
        }
    }

    public void Update_Joint_target(float new_ja)
    {
        joint_target = new_ja;
        if (gameObject.name == "Joint7")
        {
            print(joint_target);
        }
    }
}
