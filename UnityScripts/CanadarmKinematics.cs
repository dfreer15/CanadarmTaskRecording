using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CanadarmKinematics : MonoBehaviour {

    public Transform base_transform;
    Matrix4x4 initial_transform;
    
    // Initializes parameters for forward kinematics
    public void initialize_FK()
    {
        initial_transform = Matrix4x4.identity;
        Vector3 unit_vector_x = new Vector3(1, 0, 0);
        Vector3 unit_vector_y = new Vector3(0, 1, 0);
        Vector3 unit_vector_z = new Vector3(0, 0, 1);

        Quaternion initial_rotation = base_transform.rotation;

        Vector3 z_direc = initial_rotation * unit_vector_z;
        Vector3 y_direc = initial_rotation * unit_vector_y;
        Vector3 x_direc = initial_rotation * unit_vector_x;

        Vector3 initial_position = base_transform.position;

        initial_transform.SetRow(0, new Vector4(x_direc[0], y_direc[0], z_direc[0], initial_position.x));
        initial_transform.SetRow(1, new Vector4(x_direc[1], y_direc[1], z_direc[1], initial_position.y));
        initial_transform.SetRow(2, new Vector4(x_direc[2], y_direc[2], z_direc[2], initial_position.z));
    }

	
    // Performs forward kinematics using the built jacobian
    public Vector3 forward_kinematics(float[] angles)
    {
        Matrix4x4 jacobian = build_canadarm_jacobian_4joint(angles);
        Matrix4x4 final_transform = initial_transform * jacobian;
        Vector3 return_position = new Vector3(final_transform[0,3], final_transform[1,3], final_transform[2,3]);

        return return_position;
    }

    // Builds full jacobian based on below DH table
    public Matrix4x4 build_canadarm_jacobian(float[] angles)
    {
        // Standard DH Table (confirmed with MATLAB)
        // [theta d      a    alpha]
        // [0     1.75   0        0]
        // [0+90  0.9    0     pi/2]
        // [0     -0.7   0    -pi/2]
        // [0     -0.7   7.71     0]
        // [0     -0.788 7.71     pi/2]
        // [0+90  0.3    0     pi/2]
        // [0     0.65   0     pi/2]
	    
	
	// Final values were partially determined via testing
        Matrix4x4 matrix1 = calculate_dh_matrix(angles[0] - Mathf.PI / 2.0f - 0.15f, 0.0f, 0.0f, 0.0f);
        Matrix4x4 matrix2 = calculate_dh_matrix(angles[1] + Mathf.PI / 2.0f - 0.1f, 0.95f, 0.0f, (3.14159f / 2.0f));
        Matrix4x4 matrix3 = calculate_dh_matrix(angles[2], -0.67f, 0.0f, -(3.14159f / 2.0f));
        Matrix4x4 matrix4 = calculate_dh_matrix(angles[3], -0.7f, 7.7f, 0);
        Matrix4x4 matrix5 = calculate_dh_matrix(angles[4], -0.7f, 7.0f, 0);
        Matrix4x4 matrix6 = calculate_dh_matrix(angles[5] + Mathf.PI / 2.0f + 0.2f, -0.76f, 0.0f, 3.14159f / 2.0f);
        Matrix4x4 matrix7 = calculate_dh_matrix(angles[6], 1.6f, 0, 3.14159f / 2.0f);

        Matrix4x4 final_jacobian = matrix1 * matrix2;
        final_jacobian *= matrix3;
        final_jacobian *= matrix4;
        final_jacobian *= matrix5;        
	final_jacobian *= matrix6;
        final_jacobian *= matrix7;
        
        return final_jacobian;
    }

    // Builds jacobian using only first 4 joints, as 
    public Matrix4x4 build_canadarm_jacobian_4joint(float[] angles)
    {
        Matrix4x4 matrix1 = calculate_dh_matrix(angles[0] - Mathf.PI / 2.0f - 0.15f, 0.0f, 0.0f, 0.0f);
        Matrix4x4 matrix2 = calculate_dh_matrix(angles[1] + Mathf.PI / 2.0f - 0.1f, 0.95f, 0.0f, (3.14159f / 2.0f));
        Matrix4x4 matrix3 = calculate_dh_matrix(angles[2], -0.67f, 0.0f, -(3.14159f / 2.0f));
        Matrix4x4 matrix4 = calculate_dh_matrix(angles[3], -0.7f, 7.7f, 0);
        Matrix4x4 matrix5 = calculate_dh_matrix(angles[4], -0.7f, 7.0f, 0);

        Matrix4x4 final_jacobian = matrix1 * matrix2;
        final_jacobian *= matrix3;
        final_jacobian *= matrix4;
        final_jacobian *= matrix5;

        return final_jacobian;
    }

    // Calculates Denavit-Hartenburg matrix for each link of the robot
    Matrix4x4 calculate_dh_matrix(float theta, float d, float a, float alpha)
    {
        Matrix4x4 return_matrix = Matrix4x4.identity;

        return_matrix.SetRow(0, new Vector4(Mathf.Cos(theta), -Mathf.Sin(theta), 0, a));
        return_matrix.SetRow(1, new Vector4(Mathf.Sin(theta) * Mathf.Cos(alpha), Mathf.Cos(theta) * Mathf.Cos(alpha), -Mathf.Sin(alpha), -d * Mathf.Sin(alpha)));
        return_matrix.SetRow(2, new Vector4(Mathf.Sin(theta) * Mathf.Sin(alpha), Mathf.Cos(theta) * Mathf.Sin(alpha), Mathf.Cos(alpha), d * Mathf.Cos(alpha)));
        return_matrix.SetRow(3, new Vector4(0, 0, 0, 1));

        return return_matrix;
    }
}
