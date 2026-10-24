import Panzoom from "@panzoom/panzoom/panzoom.es.js";

export default {
  template: `
    <div ref="viewport" class="review-viewport"
         style="position:relative;overflow:hidden;touch-action:none"
         @wheel.prevent="onWheel" @pointerdown="onPointerDown"
         @load.capture="onImageLoaded">
      <div ref="content" style="position:absolute;transform-origin:50% 50%">
        <slot></slot>
      </div>
    </div>
  `,
  props: {
    imageKey: String,
    imageSize: { type: Array, default: () => [640, 480] },
    subject: Array,
    focusSubject: Boolean,
    interactionMode: { type: String, default: "box" },
  },
  data() {
    return { fitWidth: 0, fitHeight: 0, sentWidth: 0, sentHeight: 0 };
  },
  mounted() {
    this.panzoom = Panzoom(this.$refs.content, {
      maxScale: 12,
      minScale: 1,
      noBind: true,
      animate: false,
      cursor: "inherit",
    });
    this.moveHandler = event => this.panzoom.handleMove(event);
    this.upHandler = event => this.panzoom.handleUp(event);
    document.addEventListener("pointermove", this.moveHandler);
    document.addEventListener("pointerup", this.upHandler);
    document.addEventListener("pointercancel", this.upHandler);
    this.$refs.content.addEventListener("panzoomchange", this.publishViewport);
    this.observer = new ResizeObserver(() => this.layout());
    this.observer.observe(this.$refs.viewport);
    this.layout();
  },
  beforeUnmount() {
    this.observer.disconnect();
    document.removeEventListener("pointermove", this.moveHandler);
    document.removeEventListener("pointerup", this.upHandler);
    document.removeEventListener("pointercancel", this.upHandler);
    this.$refs.content.removeEventListener("panzoomchange", this.publishViewport);
    this.panzoom.destroy();
  },
  watch: {
    imageKey() {
      this.$nextTick(() => this.applyView());
    },
    imageSize(value, previous) {
      if (JSON.stringify(value) !== JSON.stringify(previous)) {
        this.$nextTick(() => this.layout());
      }
    },
    subject(value, previous) {
      if (JSON.stringify(value) !== JSON.stringify(previous) && this.focusSubject) {
        this.$nextTick(() => this.applyView());
      }
    },
    focusSubject() {
      this.$nextTick(() => this.applyView());
    },
  },
  methods: {
    layout() {
      const viewport = this.$refs.viewport;
      if (!viewport.clientWidth || !viewport.clientHeight) return;
      const ratio = Math.min(
        viewport.clientWidth / this.imageSize[0],
        viewport.clientHeight / this.imageSize[1],
      );
      this.fitWidth = this.imageSize[0] * ratio;
      this.fitHeight = this.imageSize[1] * ratio;
      Object.assign(this.$refs.content.style, {
        width: `${this.fitWidth}px`,
        height: `${this.fitHeight}px`,
        left: `${(viewport.clientWidth - this.fitWidth) / 2}px`,
        top: `${(viewport.clientHeight - this.fitHeight) / 2}px`,
      });
      this.applyView();
    },
    applyView() {
      if (!this.panzoom || !this.fitWidth) return;
      if (!this.focusSubject || !this.subject) {
        this.fit();
        return;
      }
      const [centerX, centerY, width, height] = this.subject;
      const scale = Math.min(12, Math.max(1, Math.min(
        this.$refs.viewport.clientWidth / (width * this.fitWidth * 1.4),
        this.$refs.viewport.clientHeight / (height * this.fitHeight * 1.4),
      )));
      this.panzoom.zoom(scale, { force: true });
      this.panzoom.pan(
        scale > 1 ? (0.5 - centerX) * this.fitWidth : 0,
        scale > 1 ? (0.5 - centerY) * this.fitHeight : 0,
        { force: true },
      );
      this.publishViewport();
    },
    fit() {
      this.panzoom.reset({ animate: false });
      this.publishViewport();
    },
    zoomIn() {
      this.panzoom.zoomIn({ animate: false });
    },
    zoomOut() {
      this.panzoom.zoomOut({ animate: false });
    },
    onWheel(event) {
      this.panzoom.zoomWithWheel(event);
    },
    onPointerDown(event) {
      if (this.interactionMode === "pan" || event.pointerType === "touch") {
        this.panzoom.handleDown(event);
      }
    },
    onImageLoaded() {
      this.$nextTick(() => this.layout());
    },
    publishViewport() {
      const scale = this.panzoom.getScale();
      const width = this.fitWidth * scale;
      const height = this.fitHeight * scale;
      if (!width || !height) return;
      if (Math.abs(width - this.sentWidth) < 0.1 && Math.abs(height - this.sentHeight) < 0.1) return;
      this.sentWidth = width;
      this.sentHeight = height;
      this.$emit("viewport", { width, height, scale });
    },
  },
};