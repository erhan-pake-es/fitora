// model_viewer_plus inserts relatedJs with innerHTML on web, so its script is
// inert. Prepare Upload and Fit Analysis previews before displaying the GLB.
(() => {
  const observedRoots = new WeakSet();
  const observedViewers = new WeakSet();
  const objectUrls = new Map();
  const observer = new MutationObserver(() => scan(document));
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  function makeBodySilver(viewer) {
    const body = viewer.model?.getMaterialByName?.('BodyMaterial') ??
      viewer.model?.getMaterialByName?.('body_gray');
    if (!body) return;
    const pbr = body.pbrMetallicRoughness;
    body.setAlphaMode('BLEND');
    pbr.setBaseColorFactor([
      0.74, 0.74, 0.74, viewer.dataset.bodyGradient === 'true' ? 1 : 0.6,
    ]);
    pbr.setMetallicFactor(0.5);
    pbr.setRoughnessFactor(0.52);
  }

  function installFitFloor(viewer) {
    const art = viewer.querySelector('.fitora-fit-floor-art');
    if (!art) return;
    const scale = () => {
      const radius = viewer.getCameraOrbit?.().radius;
      if (!Number.isFinite(radius) || radius <= 0) return;
      const value = Math.max(.35, Math.min(5,
        (viewer.clientHeight / 700) * (2.7 / radius)));
      art.style.setProperty('--fitora-floor-scale', String(value));
    };
    const anchor = () => {
      const size = viewer.getDimensions?.();
      const center = viewer.getBoundingBoxCenter?.();
      if (!size || !center) return;
      viewer.updateHotspot?.({name: 'hotspot-fit-floor',
        position: `${center.x}m ${center.y - size.y / 2}m ${center.z + size.z * .45}m`});
      scale();
    };
    viewer.addEventListener('load', anchor);
    viewer.addEventListener('camera-change', scale);
    if (window.ResizeObserver) new ResizeObserver(scale).observe(viewer);
    if (viewer.loaded) anchor();
  }

  function smoothBodyNormals(buffer) {
    if (buffer.byteLength < 28) return null;
    const source = new DataView(buffer);
    if (source.getUint32(0, true) !== 0x46546c67 ||
        source.getUint32(4, true) !== 2) return null;
    const jsonLength = source.getUint32(12, true);
    const json = JSON.parse(decoder.decode(
      new Uint8Array(buffer, 20, jsonLength),
    ));
    const binaryHeader = 20 + jsonLength;
    if (source.getUint32(binaryHeader + 4, true) !== 0x004e4942) return null;
    const binaryLength = source.getUint32(binaryHeader, true);
    const binary = new Uint8Array(buffer, binaryHeader + 8, binaryLength);
    const bodyMaterial = json.materials?.findIndex(
      (material) => material.name === 'BodyMaterial' ||
        material.name === 'body_gray',
    );
    if (bodyMaterial == null || bodyMaterial < 0) return null;

    function readAccessor(index, components) {
      const accessor = json.accessors[index];
      const view = json.bufferViews[accessor.bufferView];
      if ((view.buffer ?? 0) !== 0 || accessor.sparse) return null;
      const bytes = accessor.componentType === 5126 ||
        accessor.componentType === 5125 ? 4 :
        accessor.componentType === 5123 ? 2 : 0;
      if (!bytes) return null;
      const stride = view.byteStride ?? bytes * components;
      const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
      const data = new DataView(binary.buffer, binary.byteOffset, binaryLength);
      const values = new Array(accessor.count * components);
      for (let vertex = 0; vertex < accessor.count; vertex++) {
        for (let part = 0; part < components; part++) {
          const at = start + vertex * stride + part * bytes;
          values[vertex * components + part] = accessor.componentType === 5126
            ? data.getFloat32(at, true)
            : accessor.componentType === 5125
              ? data.getUint32(at, true) : data.getUint16(at, true);
        }
      }
      return values;
    }

    const additions = [];
    for (const mesh of json.meshes ?? []) {
      for (const primitive of mesh.primitives ?? []) {
        if (primitive.material !== bodyMaterial ||
            (primitive.mode ?? 4) !== 4) continue;
        const positions = readAccessor(primitive.attributes.POSITION, 3);
        if (!positions) continue;
        const indices = primitive.indices == null
          ? Array.from({length: positions.length / 3}, (_, i) => i)
          : readAccessor(primitive.indices, 1);
        if (!indices || indices.length % 3 !== 0) continue;
        let smooth = null;
        if (primitive.attributes.NORMAL == null) {
          const normals = new Float64Array(positions.length);
          for (let i = 0; i < indices.length; i += 3) {
            const a = indices[i] * 3;
            const b = indices[i + 1] * 3;
            const c = indices[i + 2] * 3;
            const abx = positions[b] - positions[a];
            const aby = positions[b + 1] - positions[a + 1];
            const abz = positions[b + 2] - positions[a + 2];
            const acx = positions[c] - positions[a];
            const acy = positions[c + 1] - positions[a + 1];
            const acz = positions[c + 2] - positions[a + 2];
            const nx = aby * acz - abz * acy;
            const ny = abz * acx - abx * acz;
            const nz = abx * acy - aby * acx;
            for (const at of [a, b, c]) {
              normals[at] += nx;
              normals[at + 1] += ny;
              normals[at + 2] += nz;
            }
          }

          // Coincident vertices at UV seams share one surface normal.
          const groups = new Map();
          function keyAt(i) {
            return `${Math.round(positions[i] * 1e6)},` +
              `${Math.round(positions[i + 1] * 1e6)},` +
              `${Math.round(positions[i + 2] * 1e6)}`;
          }
          for (let i = 0; i < positions.length; i += 3) {
            const key = keyAt(i);
            const group = groups.get(key) ?? [0, 0, 0];
            group[0] += normals[i];
            group[1] += normals[i + 1];
            group[2] += normals[i + 2];
            groups.set(key, group);
          }
          smooth = new Float32Array(positions.length);
          for (let i = 0; i < positions.length; i += 3) {
            const [x, y, z] = groups.get(keyAt(i));
            const length = Math.hypot(x, y, z) || 1;
            smooth[i] = x / length;
            smooth[i + 1] = y / length;
            smooth[i + 2] = z / length;
          }
        }

        const heights = positions.filter((_, i) => i % 3 === 1);
        const minY = Math.min(...heights);
        const height = Math.max(...heights) - minY;
        const fadeStart = minY + height * 0.82;
        const fadeEnd = minY + height * 0.885;
        const colors = new Float32Array(positions.length / 3 * 4);
        for (let vertex = 0; vertex < colors.length / 4; vertex++) {
          const x = positions[vertex * 3];
          const y = positions[vertex * 3 + 1];
          const heightMix = Math.max(0, Math.min(1,
            (y - fadeStart) / (fadeEnd - fadeStart),
          ));
          const centerMix = Math.max(0, Math.min(1,
            (height * 0.105 - Math.abs(x)) / (height * 0.045),
          ));
          const fade = heightMix * heightMix * (3 - 2 * heightMix) * centerMix;
          colors.set([1, 1, 1, 0.6 + 0.3 * fade], vertex * 4);
        }
        additions.push({primitive, smooth, colors});
      }
    }
    if (!additions.length) return null;

    const outputLength = binaryLength + additions.reduce(
      (sum, item) => sum + (item.smooth?.byteLength ?? 0) +
        item.colors.byteLength,
      0,
    );
    const outputBinary = new Uint8Array(outputLength);
    outputBinary.set(binary);
    let offset = binaryLength;
    for (const {primitive, smooth, colors} of additions) {
      if (smooth) {
        new Float32Array(outputBinary.buffer, offset, smooth.length).set(smooth);
        json.bufferViews.push({
          buffer: 0, byteOffset: offset, byteLength: smooth.byteLength,
          target: 34962,
        });
        json.accessors.push({
          bufferView: json.bufferViews.length - 1,
          componentType: 5126, count: smooth.length / 3, type: 'VEC3',
        });
        primitive.attributes.NORMAL = json.accessors.length - 1;
        offset += smooth.byteLength;
      }
      new Float32Array(outputBinary.buffer, offset, colors.length).set(colors);
      json.bufferViews.push({
        buffer: 0, byteOffset: offset, byteLength: colors.byteLength,
        target: 34962,
      });
      json.accessors.push({
        bufferView: json.bufferViews.length - 1,
        componentType: 5126, count: colors.length / 4, type: 'VEC4',
      });
      primitive.attributes.COLOR_0 = json.accessors.length - 1;
      offset += colors.byteLength;
    }
    json.buffers[0].byteLength = outputBinary.length;
    const jsonBytes = encoder.encode(JSON.stringify(json));
    const paddedJsonLength = (jsonBytes.length + 3) & ~3;
    const output = new ArrayBuffer(12 + 8 + paddedJsonLength + 8 + outputLength);
    const header = new DataView(output);
    header.setUint32(0, 0x46546c67, true);
    header.setUint32(4, 2, true);
    header.setUint32(8, output.byteLength, true);
    header.setUint32(12, paddedJsonLength, true);
    header.setUint32(16, 0x4e4f534a, true);
    const bytes = new Uint8Array(output);
    bytes.fill(0x20, 20, 20 + paddedJsonLength);
    bytes.set(jsonBytes, 20);
    const outputBinaryHeader = 20 + paddedJsonLength;
    header.setUint32(outputBinaryHeader, outputLength, true);
    header.setUint32(outputBinaryHeader + 4, 0x004e4942, true);
    bytes.set(outputBinary, outputBinaryHeader + 8);
    return output;
  }

  async function prepareViewer(viewer, state) {
    const originalSrc = viewer.getAttribute('src');
    if (!originalSrc) return;
    viewer.style.visibility = 'hidden';
    try {
      const response = await fetch(originalSrc);
      if (!response.ok) throw new Error(`Model request failed: ${response.status}`);
      const model = smoothBodyNormals(await response.arrayBuffer());
      if (!model || !viewer.isConnected ||
          viewer.getAttribute('src') !== originalSrc) {
        state.pending = false;
        viewer.style.visibility = 'visible';
        return;
      }
      const url = URL.createObjectURL(new Blob([model], {
        type: 'model/gltf-binary',
      }));
      objectUrls.set(viewer, url);
      state.finalSrc = url;
      viewer.dataset.bodyGradient = 'true';
      viewer.setAttribute('src', url);
    } catch (error) {
      console.warn('Silver body preview smoothing unavailable:', error);
      state.pending = false;
      viewer.style.visibility = 'visible';
    }
  }

  function watchViewer(viewer) {
    if (observedViewers.has(viewer)) return;
    observedViewers.add(viewer);
    if (viewer.id === 'fitora-fit-preview') installFitFloor(viewer);
    const state = {pending: true, finalSrc: null};
    viewer.addEventListener('load', () => {
      makeBodySilver(viewer);
      if (viewer.getAttribute('src') === state.finalSrc || !state.pending) {
        viewer.style.visibility = 'visible';
      }
    });
    if (viewer.loaded) makeBodySilver(viewer);
    void prepareViewer(viewer, state);
  }

  function scan(root) {
    if (!observedRoots.has(root)) {
      observer.observe(root, {childList: true, subtree: true});
      observedRoots.add(root);
    }
    for (const viewer of root.querySelectorAll?.(
      '#fitora-upload-preview, [id^="fitora-fit-"]',
    ) ?? []) {
      watchViewer(viewer);
    }
    for (const node of root.querySelectorAll?.('*') ?? []) {
      if (node.shadowRoot && node.localName !== 'model-viewer') {
        scan(node.shadowRoot);
      }
    }
    for (const [node, url] of objectUrls) {
      if (!node.isConnected) {
        URL.revokeObjectURL(url);
        objectUrls.delete(node);
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => scan(document), {
      once: true,
    });
  } else {
    scan(document);
  }
})();
