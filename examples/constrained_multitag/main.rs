//! Rust port of `examples/constrained_multitag/main.py`.
//!
//! Determines a robot pose from the corner pixel locations of several
//! AprilTags. The robot pose is constrained to be on the floor (`z = 0`).

use hafgufa::math::{cos, sin};
use hafgufa::{Problem, Variable, VariableArena, VariableMatrix, solve};
use ndarray::{Array2, array};

fn main() {
    // Camera calibration.
    let fx = 600.0_f64;
    let fy = 600.0_f64;
    let cx = 300.0_f64;
    let cy = 150.0_f64;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    // Robot pose.
    let robot_x = problem.decision_variable();
    let robot_y = problem.decision_variable();
    let robot_z = Variable::constant_in(&arena, 0.0);
    let robot_theta = problem.decision_variable();

    // Cache the trig variables so the expression graph shares the nodes.
    let sin_theta = sin(robot_theta);
    let cos_theta = cos(robot_theta);

    // 4×4 field-to-robot homogeneous transform.
    let zero = Variable::constant_in(&arena, 0.0);
    let one = Variable::constant_in(&arena, 1.0);
    let mut field2robot = VariableMatrix::zeros_in(&arena, 4, 4);
    field2robot.set_variable(0, 0, cos_theta);
    field2robot.set_variable(0, 1, -sin_theta);
    field2robot.set_variable(0, 2, zero);
    field2robot.set_variable(0, 3, robot_x);
    field2robot.set_variable(1, 0, sin_theta);
    field2robot.set_variable(1, 1, cos_theta);
    field2robot.set_variable(1, 2, zero);
    field2robot.set_variable(1, 3, robot_y);
    field2robot.set_variable(2, 0, zero);
    field2robot.set_variable(2, 1, zero);
    field2robot.set_variable(2, 2, one);
    field2robot.set_variable(2, 3, robot_z);
    field2robot.set_variable(3, 0, zero);
    field2robot.set_variable(3, 1, zero);
    field2robot.set_variable(3, 2, zero);
    field2robot.set_variable(3, 3, one);

    // Robot frame is ENU, camera frame is SDE.
    let robot2camera = VariableMatrix::from_array_in(
        &arena,
        &array![
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );

    let field2camera = &field2robot * &robot2camera;

    // 4×1 (x, y, z, 1) points in the field frame.
    let field2points = [
        VariableMatrix::from_array_in(
            &arena,
            &Array2::from_shape_vec((4, 1), vec![2.0, 0.0 - 0.08255, 0.4, 1.0]).unwrap(),
        ),
        VariableMatrix::from_array_in(
            &arena,
            &Array2::from_shape_vec((4, 1), vec![2.0, 0.0 + 0.08255, 0.4, 1.0]).unwrap(),
        ),
    ];

    // Hand-calibrated observations: the pixel locations we'd see for a
    // camera located at (0, 0, 0).
    let point_observations: [(f64, f64); 2] = [(325.0, 30.0), (275.0, 30.0)];

    // Initial guess — we expect the solver to converge to (0, 0, 0).
    robot_x.set_value(-0.1);
    robot_y.set_value(0.0);
    robot_theta.set_value(0.2);

    // camera2field such that field2camera * camera2field = I.
    let identity = VariableMatrix::from_array_in(&arena, &Array2::eye(4));
    let camera2field = solve(&field2camera, identity);

    // Cost: sum of squared reprojection errors.
    let mut cost = Variable::constant_in(&arena, 0.0);
    for (field2point, (u_observed, v_observed)) in
        field2points.iter().zip(point_observations.iter())
    {
        // camera2point = camera2field * field2point (4×1)
        let camera2point = &camera2field * field2point;

        let x = camera2point.get(0, 0);
        let y = camera2point.get(1, 0);
        let z = camera2point.get(2, 0);

        println!(
            "camera2point = {}, {}, {}",
            x.value(),
            y.value(),
            z.value()
        );

        let big_x = x / z;
        let big_y = y / z;

        let u = fx * big_x + cx;
        let v = fy * big_y + cy;

        println!("Expected u {}, saw {}", u.value(), u_observed);
        println!("Expected v {}, saw {}", v.value(), v_observed);

        let u_err = u - *u_observed;
        let v_err = v - *v_observed;

        cost = cost + u_err * u_err + v_err * v_err;
    }

    problem.minimize(cost);

    #[allow(unused_mut)]
    let mut opts = hafgufa::Options::default();
    #[cfg(feature = "diagnostics")]
    {
        opts = opts.diagnostics(true);
    }
    match problem.solve(opts) {
        Ok(()) => println!("exit status: success"),
        Err(e) => println!("exit status: {e}"),
    }

    println!("x = {} m", robot_x.value());
    println!("y = {} m", robot_y.value());
    println!("\u{3b8} = {} rad", robot_theta.value());
}
