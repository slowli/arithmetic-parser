//! Test for `Prototype`s.

use arithmetic_eval::{
    env::Assertions, fns, CallContext, Environment, ErrorKind, EvalResult, NativeFn, Object,
    Prototype, SpannedValue, Value,
};

use crate::evaluate;

fn as_primitive(value: &Value<'_, f32>) -> Option<f32> {
    match value {
        Value::Prim(prim) => Some(*prim),
        _ => None,
    }
}

#[derive(Debug)]
struct PointLen;

impl NativeFn<f32> for PointLen {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, f32>>,
        context: &mut CallContext<'_, 'a, f32>,
    ) -> EvalResult<'a, f32> {
        context.check_args_count(&args, 1)?;
        let point = if let Value::Object(obj) = args.pop().unwrap().extra {
            obj
        } else {
            let err = ErrorKind::native("Function argument must be an object");
            return Err(context.call_site_error(err));
        };

        let x = point.get("x").and_then(as_primitive).ok_or_else(|| {
            let err = ErrorKind::native("point must have numeric `x` field");
            context.call_site_error(err)
        })?;
        let y = point.get("y").and_then(as_primitive).ok_or_else(|| {
            let err = ErrorKind::native("point must have numeric `y` field");
            context.call_site_error(err)
        })?;
        let len = (x * x + y * y).sqrt();
        Ok(Value::Prim(len))
    }
}

#[test]
fn prototype_basics() {
    let mut proto: Object<f32> = Object::default();
    proto.insert("len", Value::native_fn(PointLen));
    let proto = Prototype::from(proto);

    let mut point: Object<f32> = vec![("x", Value::Prim(3.0)), ("y", Value::Prim(4.0))]
        .into_iter()
        .collect();
    point.set_prototype(proto.clone());

    let mut env = Environment::new();
    env.insert("point", point.into());
    let return_value = evaluate(&mut env, "point.len()");
    let return_value = as_primitive(&return_value).unwrap();
    assert!((return_value - 5.0).abs() < 1e-4, "{}", return_value);

    env.extend(Assertions.iter());
    env.insert("Point", proto.into())
        .insert_native_fn("map", fns::Map)
        .insert_wrapped_fn("abs", f32::abs);

    let program = r#"
        pt = Point(#{ x: 3, y: 4 });
        assert(abs(pt.len() - 5.0) < 0.0001);
        assert(abs((Point.len)(pt) - 5.0) < 0.0001);
        assert(abs((Point.len)(#{ x: 3, y: 4 }) - 5.0) < 0.0001);

        points_data = (#{ x: 3, y: 4 }, #{ x: -5, y: 12 });
        points = map(points_data, Point);
        assert_eq(map(points, Point.len), map(points_data, Point.len));
    "#;
    evaluate(&mut env, program);
}

#[test]
fn defining_prototype_in_script() {
    let mut env = Environment::<f32>::new();
    env.extend(Assertions.iter());
    env.insert_native_fn("impl", fns::CreatePrototype)
        .insert_wrapped_fn("abs", f32::abs);

    let program = r#"
        Manhattan = {
            len = |{x, y}| abs(x) + abs(y);
            impl(#{ len, dist: |self, other| len(self - other) })
        };
        pt = Manhattan(#{ x: 3, y: 4 });
        other_pt = Manhattan(#{ x: 2, y: -1 });
        assert_eq(pt.len(), 7);
        assert_eq(other_pt.len(), 3);
        assert_eq(pt.dist(other_pt), 6);
    "#;
    evaluate(&mut env, program);

    let proto_string = env["Manhattan"].to_string();
    assert!(proto_string.contains("prototype"), "{}", proto_string);
    assert!(
        proto_string.contains("dist: (interpreted fn @ *:4:32)"),
        "{}",
        proto_string
    );
    assert!(
        proto_string.contains("len: (interpreted fn @ *:3:19)"),
        "{}",
        proto_string
    );
}
