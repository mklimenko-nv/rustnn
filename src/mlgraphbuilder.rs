use std::collections::HashMap;

use crate::error::GraphBuilderError;
use crate::mlcontext::{MLGraph, MLOperand, MLOperandDescriptor};
use crate::{
    GraphInfo,
    mlcontext::{MLBackendBuilder, MLContext},
};
use crate::{Operand, OperandKind};

pub type Result<T> = std::result::Result<T, GraphBuilderError>;

#[derive(Debug)]
pub struct MLGraphBuilder<'context, 'builder> {
    backend: Box<dyn MLBackendBuilder<'context, 'builder> + 'builder>,

    graph: Option<GraphInfo>,
}

impl<'context, 'builder> MLGraphBuilder<'context, 'builder> {
    ///[[hasBuilt]] of type boolean
    ///
    /// Whether MLGraphBuilder.build() has been called. Once built, the MLGraphBuilder can no longer create operators or compile MLGraphs.
    pub(crate) fn has_built(&self) -> bool {
        self.graph.is_none()
    }
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
        if self.has_built() {
            panic!("Graph was already build");
        }

        // spec: If outputs is empty, then return a new promise in realm rejected with a TypeError.
        if outputs.is_empty() {
            return Err(GraphBuilderError::EmptyOutputHashMap.into());
        }

        let mut graph = self.graph.take().unwrap();

        for (name, operand) in outputs.iter() {
            // spec: If name is empty, then return a new promise in realm rejected with a TypeError.
            if name.is_empty() {
                return Err(GraphBuilderError::EmptyOutputHashMap.into());
            }

            if let Some(op) = graph.operands.get_mut(operand.id) {
                // spec: If operand is in this’s graph’s inputs or constants, then return a new promise in realm rejected with a TypeError.
                if op.kind == OperandKind::Input {
                    return Err(GraphBuilderError::RequestedInputAsOutput(op.clone()).into());
                } else if op.kind == OperandKind::Constant {
                    return Err(GraphBuilderError::RequestedConstantAsOutput(op.clone()).into());
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

        let graph = self.graph.as_mut().expect("Graph was already built");

        let id = graph.operands.len();
        graph.operands.push(operand);
        graph.input_operands.push(id as u32);

        Ok(MLOperand { id })
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
        let _graph = builder.build(&outputs).unwrap();
    }
}
