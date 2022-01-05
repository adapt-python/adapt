:ref:`{{module}} <{{module}}>`.{{objname}}
{{ underline }}====================================

.. currentmodule:: {{module}}

.. autoclass:: {{objname}}
	:no-members:
	:no-inherited-members:
	:no-special-members:
	
	
	{% block methods %}
	
	{%- set excludedmethods = [
		'add_loss',
		'add_metric',
		'add_update',
		'add_variable',
		'add_weight',
		'apply',
		'build',
		'call',
		'compute_mask',
		'compute_output_shape',
		'compute_output_signature',
		'count_params',
		'evaluate',
		'evaluate_generator',
		'finalize_state',
		'fit_generator',
		'from_config',
		'get_config',
		'get_input_at',
		'get_input_mask_at',
		'get_input_shape_at',
		'get_layer',
		'get_losses_for',
		'get_output_at',
		'get_output_mask_at',
		'get_output_shape_at',
		'get_updates_for',
		'get_weights',
		'load_weights',
		'make_predict_function',
		'make_test_function',
		'make_train_function',
		'predict_generator',
		'predict_on_batch',
		'predict_step',
		'reset_metrics',
		'reset_states',
		'save',
		'save_spec',
		'save_weights',
		'set_weights',
		'summary',
		'test_on_batch',
		'test_step',
		'to_json',
		'to_yaml',
		'train_on_batch',
		'train_step',
		'with_name_scope']
	%}
	
	.. rubric:: Methods
	
	.. autosummary::
	    {% for item in methods %}
		{%- if item not in excludedmethods %}
		~{{objname}}.{{ item }}
		{%- endif %}
		{%- endfor %}
	
	{% for item in methods %}
	{%- if item not in excludedmethods %}
	.. automethod:: {{ item }}
	{%- endif %}
	{%- endfor %}
	
    {% endblock %}
	
	
.. raw:: html

   <h2> Examples </h2>
	
.. include:: ../gallery/{{objname}}.rst
	
