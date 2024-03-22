rephrased_retriever_template = \
'''Dado el 'input' tu tarea es construir el 'nuevo_input'.
El 'nuevo_input' consta de 3 lineas mayusculas.

## Ejemplo

Un input que brinda informacion sobre las 3 lineas:

```
input = "Explicar el articulo 5 de la poliza 'Seguro para prestaciones médicas de alto costo' id POL320210210"
nuevo_input = """
POL320210210
SEGURO PARA PRESTACIONES MÉDICAS DE ALTO COSTO
ARTÍCULO 5º:
"""
```

Un 'input' que brinda información sobre 2 lineas:

```
input = "Buscar los articulos del codigo de comercio que se mencionan en el 'articulo 13: solucion de controversias' de la poliza POL320210210"
nuevo_input = """
POL320210210

ARTÍCULO 13º: SOLUCIÓN DE CONTROVERSIAS
"""
```

## Real

input: {input}
nuevo_input:
'''