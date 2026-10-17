import { createReferenceAssets } from './portal-reference.mjs';

const layer = document.getElementById('portal-flight');
const canvas = document.getElementById('portal-tunnel');
const skip = document.getElementById('skip-portal');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const DURATION = 2800;
let resources = null;
let unavailable = false;
let playing = null;
let finishActive = null;

function prepare() {
  if (resources || unavailable || motion.matches || !window.THREE) return resources;
  const THREE = window.THREE;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'default' });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, innerWidth < 600 ? 1.4 : 1.75));
    renderer.setSize(innerWidth, innerHeight);
    renderer.setClearColor(0x160a11, 1);
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x160a11, .035);
    const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, .1, 700);
    const reference = createReferenceAssets(THREE);
    const path = new THREE.CatmullRomCurve3(reference.createCircularPath());
    const geometry = new THREE.TubeBufferGeometry(path, innerWidth < 600 ? 420 : 700, 3, 12, false);
    const colors = [];
    for (let i = 0; i < geometry.attributes.position.count; i++) {
      const color = new THREE.Color(i % 2 === 0 ? '#fc9385' : '#a31551');
      colors.push(color.r, color.g, color.b);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    scene.add(new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({
      side: THREE.BackSide, vertexColors: true, wireframe: true,
      emissive: 0x4a0221, emissiveIntensity: .65,
    })));
    scene.add(new THREE.AmbientLight(0x99516d, .8));
    const lights = [0xffffff, 0xfc9385, 0xa31551, 0xfc9385, 0xa31551].map((color) => {
      const light = new THREE.PointLight(color, 1.5, 45);
      scene.add(light);
      return light;
    });
    const starTexture = new THREE.CanvasTexture(reference.createCircleTexture());
    const starPositions = new Float32Array((innerWidth < 600 ? 1200 : 3500) * 3);
    for (let i = 0; i < starPositions.length; i++) starPositions[i] = THREE.MathUtils.randFloatSpread(650);
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    scene.add(new THREE.Points(starGeometry, new THREE.PointsMaterial({
      color: 0xffd6e5, size: .75, map: starTexture, transparent: true, opacity: .65, depthWrite: false,
    })));
    const spriteCount = innerWidth < 600 ? 12 : 28;
    for (let i = 0; i < spriteCount; i++) {
      const sprite = reference.createCodeSnippetSprite(reference.snippetVarieties[i % reference.snippetVarieties.length]);
      const progress = .006 + (i / spriteCount) * .21;
      const point = path.getPointAt(progress);
      const direction = path.getTangentAt(progress);
      const side = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0)).normalize();
      sprite.position.copy(point).addScaledVector(side, i % 2 ? 3.8 : -3.8);
      sprite.position.y += Math.sin(i * 2.1) * 2.5;
      sprite.scale.multiplyScalar(.16);
      sprite.material.opacity = .55;
      scene.add(sprite);
    }
    camera.position.copy(path.getPointAt(.001));
    camera.lookAt(path.getPointAt(.006));
    renderer.compile(scene, camera);
    // Upload generated textures before the visible flight starts.
    renderer.render(scene, camera);
    resources = { THREE, renderer, scene, camera, path, lights };
    return resources;
  } catch {
    unavailable = true;
    renderer?.dispose();
    return null;
  }
}

function dispose() {
  if (!resources) return;
  const disposed = new Set();
  resources.scene.traverse((object) => {
    object.geometry?.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      if (!material) continue;
      if (material.map && !disposed.has(material.map)) { material.map.dispose(); disposed.add(material.map); }
      material.dispose();
    }
  });
  resources.renderer.dispose();
  resources = null;
}

function play() {
  if (playing) return playing;
  if (motion.matches) return Promise.resolve();
  const state = prepare();
  playing = new Promise((resolve) => {
    let frame = null;
    let closing = false;
    let done = false;
    let endTimer = null;
    const started = performance.now();
    const cleanup = () => {
      if (done) return;
      done = true;
      cancelAnimationFrame(frame);
      clearTimeout(watchdog);
      clearTimeout(endTimer);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', visibility);
      motion.removeEventListener('change', finish);
      canvas.removeEventListener('webglcontextlost', contextLost);
      skip.removeEventListener('click', finish);
      finishActive = null;
      dispose();
      resolve();
    };
    const finish = () => {
      if (closing) return;
      closing = true;
      cancelAnimationFrame(frame);
      layer.dataset.phase = 'leaving';
      layer.classList.add('portal-flight-leaving');
      document.getElementById('portal-flight-message').textContent = 'Добро пожаловать в Forte Vision';
      endTimer = setTimeout(cleanup, motion.matches ? 0 : 380);
    };
    const resize = () => {
      if (!state) return;
      state.camera.aspect = innerWidth / innerHeight;
      state.camera.updateProjectionMatrix();
      state.renderer.setSize(innerWidth, innerHeight);
    };
    const visibility = () => { if (document.hidden) finish(); };
    const contextLost = (event) => { event.preventDefault(); finish(); };
    const watchdog = setTimeout(finish, DURATION + 1000);
    finishActive = cleanup;
    skip.addEventListener('click', finish);
    motion.addEventListener('change', finish);
    canvas.addEventListener('webglcontextlost', contextLost);
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('resize', resize);
    document.querySelectorAll('.auth-header, .auth-main, .auth-footer').forEach((element) => { element.inert = true; });
    layer.hidden = false;
    layer.dataset.phase = 'opening';
    layer.classList.toggle('portal-flight-fallback', !state);
    // Commit the hidden state before starting the opacity transition.
    layer.getBoundingClientRect();
    layer.classList.add('is-active');
    skip.focus({ preventScroll: true });

    const render = (now) => {
      if (closing) return;
      const elapsed = now - started;
      const t = Math.min(elapsed / DURATION, 1);
      layer.dataset.phase = elapsed < 500 ? 'opening' : 'travel';
      if (state) {
        try {
          const eased = t * t * (3 - 2 * t);
          const pct = .001 + eased * .14;
          const { camera, path, renderer, scene, lights } = state;
          camera.position.copy(path.getPointAt(pct));
          camera.lookAt(path.getPointAt(Math.min(pct + .006, .99)));
          lights.forEach((light, i) => light.position.copy(path.getPointAt(Math.min(pct + .004 + i * .012, .99))));
          renderer.render(scene, camera);
        } catch { finish(); return; }
      }
      if (t >= 1 || (!state && elapsed >= 600)) finish();
      else frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
  });
  return playing;
}

window.FortePortal = { play };
const warmUp = () => { if (!document.hidden && !motion.matches) prepare(); };
if ('requestIdleCallback' in window) requestIdleCallback(warmUp, { timeout: 1200 });
else setTimeout(warmUp, 400);
window.addEventListener('pagehide', () => { if (finishActive) finishActive(); else dispose(); });
