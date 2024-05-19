

(
// definimos el synth
SynthDef(\movement, {
	| x = 0|
	var freq = x.linexp(0, 640, 400, 800); // mapeamos el eje x a frecuencia 400 - 800
	var amp = 0.8; // mapeamos la amplitud 0 - 0.8 (pueden modificarlo a 1, pero sonará más fuerte)
	var osc = SinOsc.ar(freq) * amp; // creamos un oscilador muy básico
	Out.ar(0, osc ! 2); // salida del audio
}).add;

//creamos el synth

x = Synth(\movement);

// iniciamos el receptor por protocolo OSC

o = OSCFunc({ |msg|
	x.set(\x, msg[1]); //actualizamos los argumentos declarados en el Synth
},  '/movement');

)
