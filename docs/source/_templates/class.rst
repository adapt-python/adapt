:ref:`{{module}} <{{module}}>`.{{objname}}
{{ underline }}====================================

.. currentmodule:: {{module}}

.. autoclass:: {{objname}}
	:no-members:
	:no-inherited-members:
	:no-special-members:
	
	
	{% block methods %}
	
	.. rubric:: Methods
	
	.. autosummary::
	    {% for item in methods %}
		~{{objname}}.{{ item }}
		{%- endfor %}
	
	{% for item in methods %}
	.. automethod:: {{ item }}
	{%- endfor %}
	
    {% endblock %}
	
	
.. raw:: html

   <h2> Examples </h2>
	
.. include:: ../gallery/{{objname}}.rst
	
