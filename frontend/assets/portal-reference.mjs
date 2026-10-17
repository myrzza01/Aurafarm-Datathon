/* Adapted from Cyberspace Portal by Matt Cannon.
 * Reference: https://codepen.io/matt-cannon/pen/OPPJLNa
 * The supplied path, particles and code sprites are retained for the login flight.
 */
export function createReferenceAssets(THREE) {
function createCircularPath() {
	const points = [];
	const totalPoints = 200;
	const controlPoints = [
		new THREE.Vector3(0, 0, 0),
		new THREE.Vector3(20, 10, -50),
		new THREE.Vector3(40, -10, -100),
		new THREE.Vector3(60, 15, -150),
		new THREE.Vector3(50, -5, -200),
		new THREE.Vector3(0, 0, -250),
		new THREE.Vector3(-100, 0, -200),
		new THREE.Vector3(-150, 0, -100),
		new THREE.Vector3(-100, 0, 0),
		new THREE.Vector3(-50, 10, 100),
		new THREE.Vector3(-20, -10, 150),
		new THREE.Vector3(0, 0, 200)
	];
	const curve = new THREE.CatmullRomCurve3(controlPoints);
	curve.tension = 0.1;
	for (let i = 0; i < totalPoints; i++) {
		const t = i / (totalPoints - 1),
			point = curve.getPoint(t);
		points.push(point);
	}
	return points;
}

function createCircleTexture() {
	const canvas = document.createElement("canvas");
	canvas.width = 32;
	canvas.height = 32;
	const context = canvas.getContext("2d");

	// Draw a circle
	context.beginPath();
	context.arc(16, 16, 16, 0, 2 * Math.PI, false);
	context.fillStyle = "white";
	context.fill();

	// Add a soft glow effect
	const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
	gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
	gradient.addColorStop(0.5, "rgba(255, 255, 255, 0.5)");
	gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

	context.globalCompositeOperation = "source-over";
	context.fillStyle = gradient;
	context.beginPath();
	context.arc(16, 16, 16, 0, 2 * Math.PI, false);
	context.fill();

	return canvas;
}


function createCodeSnippetSprite(text) {
	const canvas = document.createElement("canvas");
	canvas.width = 400;
	canvas.height = 250;
	const ctx = canvas.getContext("2d");

	// Fully transparent background, no border
	ctx.clearRect(0, 0, canvas.width, canvas.height);

	// Syntax highlighting colors from popular themes
	const colors = {
		keyword: "#ed88b1", // pink
		string: "#f1fa8c", // yellow
		comment: "#b16d8c", // blue-grey
		function: "#fc9385", // green
		variable: "#ffd0df", // cyan
		tag: "#ed88b1", // pink
		attribute: "#fc9385" // green
	};

	ctx.font = "20px 'Consolas', monospace";
	ctx.textAlign = "left";
	ctx.textBaseline = "top";

	const lines = text.split("\n");

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		let xPosition = 15;

		// Extremely simple syntax highlighting
		if (
			line.includes("const ") ||
			line.includes("function ") ||
			line.includes("if(") ||
			line.includes("return")
		) {
			// Keywords and flow control
			const parts = line.split(/\b/);
			for (const part of parts) {
				if (
					[
						"const",
						"function",
						"return",
						"if",
						"class",
						"=>",
						"import",
						"export"
					].includes(part)
				) {
					ctx.fillStyle = colors.keyword;
				} else if (part.startsWith('"') || part.startsWith("'")) {
					ctx.fillStyle = colors.string;
				} else if (part.startsWith("//")) {
					ctx.fillStyle = colors.comment;
				} else if (part.match(/^[a-zA-Z_][a-zA-Z0-9_]*\(/)) {
					ctx.fillStyle = colors.function;
				} else if (part.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/)) {
					ctx.fillStyle = colors.variable;
				} else {
					ctx.fillStyle = "#f8f8f2"; // default text color
				}

				const width = ctx.measureText(part).width;
				ctx.fillText(part, xPosition, 15 + i * 24);
				xPosition += width;
			}
		} else if (line.includes("<") && line.includes(">")) {
			// HTML-like syntax
			const parts = line.split(/(<\/?[a-zA-Z0-9-]+|>|="[^"]*")/g);
			for (const part of parts) {
				if (part.startsWith("<") && !part.startsWith("</")) {
					ctx.fillStyle = colors.tag;
				} else if (part.startsWith("</") || part === ">") {
					ctx.fillStyle = colors.tag;
				} else if (part.startsWith("=")) {
					ctx.fillStyle = colors.attribute;
				} else if (part.startsWith('"')) {
					ctx.fillStyle = colors.string;
				} else {
					ctx.fillStyle = "#f8f8f2"; // default text color
				}

				const width = ctx.measureText(part).width;
				ctx.fillText(part, xPosition, 15 + i * 24);
				xPosition += width;
			}
		} else if (line.includes("{") || line.includes("}") || line.includes(";")) {
			// CSS-like syntax
			ctx.fillStyle = "#f8f8f2"; // default for CSS
			ctx.fillText(line, xPosition, 15 + i * 24);
		} else {
			// Default rendering
			ctx.fillStyle = "#f8f8f2";
			ctx.fillText(line, 15, 15 + i * 24);
		}
	}

	const texture = new THREE.CanvasTexture(canvas);
	texture.minFilter = THREE.LinearFilter;
	const material = new THREE.SpriteMaterial({
		map: texture,
		transparent: true,
		opacity: 0.8,
		blending: THREE.AdditiveBlending
	});

	const sprite = new THREE.Sprite(material);

	// Randomize scale for variety
	let scaleFactor = 8 + Math.random() * 12;
	sprite.scale.set(scaleFactor, scaleFactor * (canvas.height / canvas.width), 1);

	return sprite;
}

const snippetVarieties = [
		// HTML Card Snippet
		[
			'<div class="card">',
			"  <h1>Let it Glow</h1>",
			"  <p>With a little bit of CSS light.</p>",
			"</div>"
		].join("\n"),

		// CSS Glow Snippet
		[
			".card {",
			"  background-color: #25121e;",
			"  border-radius: 12px;",
			"  box-shadow: 0 8px 20px -4px #fc9385;",
			"}"
		].join("\n"),

		// Advanced Glow CSS
		[
			".glow-card {",
			"  box-shadow:",
			"    0 0 10px rgba(163, 21, 81, 0.5),",
			"    0 0 20px rgba(163, 21, 81, 0.3),",
			"    inset 0 0 10px rgba(163, 21, 81, 0.2);",
			"}"
		].join("\n"),

		// JavaScript Interaction
		[
			"document.querySelector('.card').addEventListener('mousemove', (e) => {",
			"  const { x, y } = e;",
			"  updateGlowPosition(x, y);",
			"});"
		].join("\n"),

		// Card Animation
		[
			"@keyframes pulse-glow {",
			"  0% { box-shadow: 0 0 10px #a31551; }",
			"  50% { box-shadow: 0 0 30px #fc9385; }",
			"  100% { box-shadow: 0 0 10px #a31551; }",
			"}"
		].join("\n"),

		// Reactive Glow Function
		[
			"function createDirectionalGlow(event, element) {",
			"  const rect = element.getBoundingClientRect();",
			"  const x = event.clientX - rect.left;",
			"  const y = event.clientY - rect.top;",
			"  // Set glow position based on cursor",
			"}"
		].join("\n"),

		// SVG Filter Glow
		[
			'<filter id="glow">',
			'  <feGaussianBlur stdDeviation="5" result="blur"/>',
			'  <feColorMatrix in="blur" values="0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 15 0"/>',
			"</filter>"
		].join("\n"),

		// CSS Variables for Glow
		[
			":root {",
			"  --glow-color: #a31551;",
			"  --glow-spread: 8px;",
			"  --glow-opacity: 0.7;",
			"}"
		].join("\n"),

		// Card Hover Effect
		[
			".card:hover {",
			"  box-shadow:",
			"    0 0 15px rgba(163, 21, 81, 0.8),",
			"    0 0 30px rgba(163, 21, 81, 0.4);",
			"  transition: box-shadow 0.3s ease;",
			"}"
		].join("\n")
	];

return { createCircularPath, createCircleTexture, createCodeSnippetSprite, snippetVarieties };
}
