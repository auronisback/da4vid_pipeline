import unittest

from da4vid.model.samples import SampleSet
from da4vid.pipeline.steps import PipelineStep, PipelineException


class StepCallbackTest(unittest.TestCase):

  class __DummySuccessfulStep(PipelineStep):

    def __init__(self, **kwargs):
      super().__init__('dummy_successful', folder='', **kwargs)

    def _execute(self, sample_set: SampleSet) -> SampleSet:
      pass

    def _resume(self, sample_set: SampleSet) -> SampleSet:
      pass

    def output_folder(self) -> str:
      pass

    def input_folder(self) -> str:
      pass

  class __DummyFailingStep(PipelineStep):

    def __init__(self, **kwargs):
      super().__init__('dummy_failing', folder='', **kwargs)

    def _execute(self, sample_set: SampleSet) -> SampleSet:
      raise PipelineException('Execution failed')

    def _resume(self, sample_set: SampleSet) -> SampleSet:
      raise PipelineException('Resuming failed')

    def output_folder(self) -> str:
      pass

    def input_folder(self) -> str:
      pass

  def test_step_execute_without_callbacks(self):
    step = StepCallbackTest.__DummySuccessfulStep()
    step.execute(None)

  def test_execute_pre_step_callback_with_single_function(self):
    called: bool = False

    def pre_fn(_: StepCallbackTest.__DummySuccessfulStep):
      nonlocal called
      called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=pre_fn
    )
    step.execute(None)
    self.assertTrue(called)

  def test_execute_pre_step_callback_with_multiple_functions(self):
    called_fn1 = False
    called_fn2 = False

    def pre_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def pre_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=[pre_fn_1, pre_fn_2]
    )
    step.execute(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_execute_after_registering_pre_step_callbacks(self):
    called_fn1 = False
    called_fn2 = False

    def pre_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def pre_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep()
    step.register_pre_step_fn(pre_fn_1)
    step.register_pre_step_fn(pre_fn_2)
    step.execute(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_execute_post_step_callback_with_single_function(self):
    called: bool = False

    def post_fn(_: StepCallbackTest.__DummySuccessfulStep):
      nonlocal called
      called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=post_fn
    )
    step.execute(None)
    self.assertTrue(called)

  def test_execute_post_step_callback_with_multiple_functions(self):
    called_fn1 = False
    called_fn2 = False

    def post_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def post_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=[post_fn_1, post_fn_2]
    )
    step.execute(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_execute_after_registering_post_step_callbacks(self):
    called_fn1 = False
    called_fn2 = False

    def post_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def post_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep()
    step.register_post_step_fn([post_fn_1, post_fn_2])
    step.execute(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_excute_step_with_pre_and_post_callbacks(self):
    pre_called = False
    post_called = False

    def pre_fn(_):
      nonlocal pre_called
      pre_called = True

    def post_fn(_):
      nonlocal post_called
      post_called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=pre_fn,
      post_step_fn=post_fn
    )
    step.execute(None)
    self.assertTrue(pre_called, 'Pre-step callback not called')
    self.assertTrue(post_called, 'Post-step callback not called')

  def test_execute_step_eith_failure_callbacks(self):
    called = False

    def fail_fn(_, e: PipelineException):
      nonlocal called
      self.assertEqual('Execution failed', e.message, 'Invalid exception')
      called = True

    step = StepCallbackTest.__DummyFailingStep(
      failed_step_fn=fail_fn
    )
    with self.assertRaises(PipelineException):
      step.execute(None)
    self.assertTrue(called, 'Failure callback has not been called')

  def test_execute_step_with_pre_and_failure_callbacks(self):
    pre_called = False
    failed_called = False

    def pre_fn(_):
      nonlocal pre_called
      pre_called = True

    def fail_fn(_, e: PipelineException):
      nonlocal failed_called
      self.assertEqual('Execution failed', e.message, 'Invalid exception')
      failed_called = True

    step = StepCallbackTest.__DummyFailingStep(
      pre_step_fn=pre_fn,
      failed_step_fn=fail_fn
    )
    with self.assertRaises(PipelineException):
      step.execute(None)
      self.assertTrue(pre_called, 'Pre-step callback not called')
      self.assertTrue(failed_called, 'Failed-step callback not called')

  def test_resume_step_without_callbacks(self):
    step = StepCallbackTest.__DummySuccessfulStep()
    step.resume(None)

  def test_resume_pre_step_callback_with_single_function(self):
    called: bool = False

    def pre_fn(_: StepCallbackTest.__DummySuccessfulStep):
      nonlocal called
      called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=pre_fn
    )
    step.resume(None)
    self.assertTrue(called)

  def test_resume_pre_step_callback_with_multiple_functions(self):
    called_fn1 = False
    called_fn2 = False

    def pre_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def pre_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=[pre_fn_1, pre_fn_2]
    )
    step.resume(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_resume_step_after_registering_pre_step_callbacks(self):
    called_fn1 = False
    called_fn2 = False

    def pre_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def pre_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep()
    step.register_pre_step_fn(pre_fn_1)
    step.register_pre_step_fn(pre_fn_2)
    step.resume(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_resume_post_step_callback_with_single_function(self):
    called: bool = False

    def post_fn(_: StepCallbackTest.__DummySuccessfulStep):
      nonlocal called
      called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=post_fn
    )
    step.resume(None)
    self.assertTrue(called)

  def test_resume_post_step_callback_with_multiple_functions(self):
    called_fn1 = False
    called_fn2 = False

    def post_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def post_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=[post_fn_1, post_fn_2]
    )
    step.resume(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_resume_ste_after_registering_post_step_callbacks(self):
    called_fn1 = False
    called_fn2 = False

    def post_fn_1(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn1
      called_fn1 = True

    def post_fn_2(_: StepCallbackTest.__DummyFailingStep):
      nonlocal called_fn2
      called_fn2 = True

    step = StepCallbackTest.__DummySuccessfulStep()
    step.register_post_step_fn([post_fn_1, post_fn_2])
    step.resume(None)
    self.assertTrue(called_fn1, 'fn1 not called')
    self.assertTrue(called_fn2, 'fn2 not called')

  def test_resume_step_with_pre_and_post_callbacks_executed(self):
    pre_called = False
    post_called = False

    def pre_fn(_):
      nonlocal pre_called
      pre_called = True

    def post_fn(_):
      nonlocal post_called
      post_called = True

    step = StepCallbackTest.__DummySuccessfulStep(
      pre_step_fn=pre_fn,
      post_step_fn=post_fn
    )
    step.resume(None)
    self.assertTrue(pre_called, 'Pre-step callback not called')
    self.assertTrue(post_called, 'Post-step callback not called')

  def test_resume_step_failure_callbacks(self):
    called = False

    def fail_fn(_, e: PipelineException):
      nonlocal called
      self.assertEqual('Resuming failed', e.message, 'Invalid exception')
      called = True

    step = StepCallbackTest.__DummyFailingStep(
      failed_step_fn=fail_fn
    )
    with self.assertRaises(PipelineException):
      step.resume(None)
      self.assertTrue(called, 'Failure callback has not been called')

  def test_resume_step_with_pre_and_failure_callbacks(self):
    pre_called = False
    failed_called = False

    def pre_fn(_):
      nonlocal pre_called
      pre_called = True

    def fail_fn(_, e: PipelineException):
      nonlocal failed_called
      self.assertEqual('Resuming failed', e.message, 'Invalid exception')
      failed_called = True

    step = StepCallbackTest.__DummyFailingStep(
      pre_step_fn=pre_fn,
      failed_step_fn=fail_fn
    )
    with self.assertRaises(PipelineException):
      step.resume(None)
      self.assertTrue(pre_called, 'Pre-step callback not called')
      self.assertTrue(failed_called, 'Failed-step callback not called')




