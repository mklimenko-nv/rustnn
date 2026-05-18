use std::collections::HashMap;

use crate::error::GraphBuilderError;
use crate::mlcontext::{MLGraph, MLOperand, MLOperandDescriptor};
use crate::operator_options::MLOperatorOptions;
use crate::{
    GraphInfo,
    mlcontext::{MLBackendBuilder, MLContext},
};
use crate::{Operand, OperandDescriptor, OperandKind, Operation};

pub type Result<T> = std::result::Result<T, GraphBuilderError>;

#[derive(Debug)]
pub struct MLGraphBuilder<'context, 'builder> {
    backend: Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>,

    graph: Option<GraphInfo>,
}

fn same_shape(inputs: &[MLOperand], graph: &GraphInfo) -> Result<OperandDescriptor> {
    let input_operand = graph
        .operands
        .get(inputs[0].id)
        .ok_or(GraphBuilderError::InvalidOperand)?;
    Ok(input_operand.descriptor.clone())
}

impl<'context, 'builder> MLGraphBuilder<'context, 'builder> {
    pub fn new(context: &'_ mut MLContext<'context, 'builder>) -> crate::error::Result<Self> {
        let backend = context.backend.create_builder()?;
        Ok(Self {
            backend,
            graph: Some(Default::default()),
        })
    }

    pub fn build_graph_info(
        &mut self,
        graph: GraphInfo,
    ) -> crate::error::Result<MLGraph<'context>> {
        self.backend.build(graph)
    }

    /*async*/
    pub fn build(
        &mut self,
        outputs: &'_ HashMap<&str, MLOperand>,
    ) -> crate::error::Result<MLGraph<'context>> {
        // spec: If outputs is empty, then return a new promise in realm rejected with a TypeError.
        if outputs.is_empty() {
            return Err(GraphBuilderError::EmptyOutputHashMap.into());
        }

        let mut graph = self
            .graph
            .take()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        for (name, operand) in outputs.iter() {
            // spec: If name is empty, then return a new promise in realm rejected with a TypeError.
            if name.is_empty() {
                return Err(GraphBuilderError::EmptyOutputHashMap.into());
            }

            if let Some(op) = graph.operands.get_mut(operand.id) {
                // spec: If operand is in this’s graph’s inputs or constants, then return a new promise in realm rejected with a TypeError.
                if op.kind == OperandKind::Input {
                    return Err(GraphBuilderError::RequestedInputAsOutput {
                        operand: op.clone(),
                        id: operand.id,
                    }
                    .into());
                } else if op.kind == OperandKind::Constant {
                    return Err(GraphBuilderError::RequestedConstantAsOutput {
                        operand: op.clone(),
                        id: operand.id,
                    }
                    .into());
                }
                op.kind = OperandKind::Output;
                op.name = Some(name.to_string());
            }
            graph.output_operands.push(operand.id as u32);
        }

        self.backend.build(graph)
    }

    pub fn input(
        &mut self,
        name: &str,
        descriptor: &MLOperandDescriptor,
    ) -> crate::error::Result<MLOperand> {
        let operand = Operand {
            descriptor: descriptor.into(),
            kind: OperandKind::Input,
            name: Some(name.to_string()),
        };

        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let id = graph.operands.len();
        graph.operands.push(operand);
        graph.input_operands.push(id as u32);

        Ok(MLOperand { id })
    }

    pub fn identity(&mut self, operand: MLOperand) -> Result<MLOperand> {
        self.identity_with_options(operand, MLOperatorOptions::default())
    }

    pub fn identity_with_options(
        &mut self,
        input: MLOperand,
        options: MLOperatorOptions,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let output_id = graph.operands.len();
        self.add_single_output_operation(
            &[input],
            Operation::Identity {
                input: input.id as u32,
                options: Some(options),
                outputs: vec![output_id as u32],
            },
            same_shape,
        )?;

        Ok(MLOperand { id: output_id })
    }

    pub fn add(&mut self, a: MLOperand, b: MLOperand) -> Result<MLOperand> {
        self.add_with_options(a, b, MLOperatorOptions::default())
    }

    pub fn add_with_options(
        &mut self,
        a: MLOperand,
        b: MLOperand,
        options: MLOperatorOptions,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;

        let output_id = graph.operands.len();
        self.add_single_output_operation(
            &[a, b],
            Operation::Add {
                a: a.id as u32,
                b: b.id as u32,
                options: Some(options),
                outputs: vec![output_id as u32],
            },
            same_shape,
        )?;

        Ok(MLOperand { id: output_id })
    }

    fn add_single_output_operation(
        &mut self,
        inputs: &[MLOperand],
        operation: Operation,
        shape_inference: impl Fn(&[MLOperand], &GraphInfo) -> Result<OperandDescriptor>,
    ) -> Result<MLOperand> {
        let graph = self
            .graph
            .as_mut()
            .ok_or(GraphBuilderError::GraphAlreadyBuilt)?;
        let output_id = graph.operands.len();

        let output_operand = Operand {
            kind: OperandKind::Output,
            descriptor: shape_inference(inputs, graph)?,
            name: operation
                .label()
                .is_empty()
                .then(|| operation.label().to_string()),
        };
        graph.operands.push(output_operand);
        graph.operations.push(operation);

        Ok(MLOperand { id: output_id })
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::{
        mlcontext::{MLContext, MLContextOptions, MLOperandDescriptor, MLPowerPreference},
        mlgraphbuilder::MLGraphBuilder,
    };

    #[test]
    fn add_inputs() {
        let _ = pretty_env_logger::try_init();
        let context = MLContext::create(&MLContextOptions {
            power_preference: MLPowerPreference::Default,
            accelerated: true,
        });
        if matches!(context, Err(crate::error::Error::NoBackendAvialable)) {
            return;
        };

        let mut context = context.unwrap();
        dbg!(&context);
        let desc = MLOperandDescriptor::new(
            crate::operator_enums::MLOperandDataType::Float32,
            [2, 2].to_vec(),
        );

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();

        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        assert_eq!(builder.graph.as_ref().unwrap().operands.len(), 2);

        let mut outputs = HashMap::new();
        outputs.insert("out1", a);
        outputs.insert("out2", b);
        let error = builder.build(&outputs).unwrap_err();
        assert!(matches!(
            error,
            crate::error::Error::GraphBuilderError { .. }
        ));
        let error_message = format!("{error}");
        dbg!(&error_message);
        assert!(error_message.contains("requested an MLOperand with id "));
        assert!(error_message.contains(" as an output that is already an input"));

        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let a = builder.input("a", &desc).unwrap();
        let b = builder.input("b", &desc).unwrap();
        assert_eq!(builder.graph.as_ref().unwrap().operands.len(), 2);
        let out1 = builder.identity(a).unwrap();
        let out2 = builder.add(a, b).unwrap();
        let mut outputs = HashMap::new();
        outputs.insert("out1", out1);
        outputs.insert("out2", out2);
        builder.build(&outputs).unwrap();
    }
}
