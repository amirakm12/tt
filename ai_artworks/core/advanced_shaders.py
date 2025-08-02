"""
Advanced GPU Shaders
High-performance shaders for real-time image processing
"""

from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from PySide6.QtGui import QOpenGLShader, QOpenGLShaderProgram
import OpenGL.GL as GL


@dataclass
class ShaderEffect:
    """Shader effect configuration"""
    name: str
    vertex_source: str
    fragment_source: str
    uniforms: Dict[str, any]
    description: str


class ShaderLibrary:
    """Collection of optimized shaders"""
    
    # Base vertex shader for all effects
    BASE_VERTEX_SHADER = """
    #version 330 core
    
    layout(location = 0) in vec3 aPosition;
    layout(location = 1) in vec2 aTexCoord;
    
    out vec2 vTexCoord;
    out vec2 vPosition;
    
    uniform mat4 uMVPMatrix;
    uniform vec2 uResolution;
    
    void main() {
        gl_Position = uMVPMatrix * vec4(aPosition, 1.0);
        vTexCoord = aTexCoord;
        vPosition = aPosition.xy;
    }
    """
    
    # Neural Glow Effect
    NEURAL_GLOW_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    in vec2 vPosition;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform float uTime;
    uniform float uIntensity;
    uniform vec3 uGlowColor;
    uniform vec2 uResolution;
    
    #define PI 3.14159265359
    
    // Fast gaussian blur approximation
    vec4 blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
        vec4 color = vec4(0.0);
        vec2 off1 = vec2(1.411764705882353) * direction;
        vec2 off2 = vec2(3.2941176470588234) * direction;
        vec2 off3 = vec2(5.176470588235294) * direction;
        color += texture(image, uv) * 0.1964825501511404;
        color += texture(image, uv + (off1 / resolution)) * 0.2969069646728344;
        color += texture(image, uv - (off1 / resolution)) * 0.2969069646728344;
        color += texture(image, uv + (off2 / resolution)) * 0.09447039785044732;
        color += texture(image, uv - (off2 / resolution)) * 0.09447039785044732;
        color += texture(image, uv + (off3 / resolution)) * 0.010381362401148057;
        color += texture(image, uv - (off3 / resolution)) * 0.010381362401148057;
        return color;
    }
    
    void main() {
        vec4 texColor = texture(uTexture, vTexCoord);
        
        // Extract bright areas
        float brightness = dot(texColor.rgb, vec3(0.2126, 0.7152, 0.0722));
        vec4 brightColor = texColor * smoothstep(0.5, 0.8, brightness);
        
        // Multi-pass blur for glow
        vec4 glow = blur13(uTexture, vTexCoord, uResolution, vec2(1.0, 0.0));
        glow = blur13(uTexture, vTexCoord, uResolution, vec2(0.0, 1.0));
        
        // Animated neural pulse
        float pulse = sin(uTime * 2.0 + length(vPosition) * 3.0) * 0.5 + 0.5;
        glow.rgb *= uGlowColor * (1.0 + pulse * 0.3);
        
        // Combine original with glow
        FragColor = texColor + glow * uIntensity * brightColor.a;
        FragColor.a = texColor.a;
    }
    """
    
    # Chromatic Aberration Effect
    CHROMATIC_ABERRATION_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform float uAmount;
    uniform vec2 uDirection;
    uniform vec2 uResolution;
    
    void main() {
        vec2 offset = uDirection * uAmount / uResolution;
        
        float r = texture(uTexture, vTexCoord - offset).r;
        float g = texture(uTexture, vTexCoord).g;
        float b = texture(uTexture, vTexCoord + offset).b;
        float a = texture(uTexture, vTexCoord).a;
        
        FragColor = vec4(r, g, b, a);
    }
    """
    
    # Advanced Blur with DOF
    DEPTH_OF_FIELD_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform sampler2D uDepthTexture;
    uniform float uFocalDepth;
    uniform float uFocalRange;
    uniform float uBlurAmount;
    uniform vec2 uResolution;
    
    const int SAMPLES = 32;
    const float GOLDEN_ANGLE = 2.39996323;
    
    vec4 depthOfField(vec2 uv) {
        float depth = texture(uDepthTexture, uv).r;
        float blur = abs(depth - uFocalDepth) / uFocalRange;
        blur = clamp(blur * uBlurAmount, 0.0, 1.0);
        
        vec4 color = texture(uTexture, uv);
        float totalWeight = 1.0;
        
        // Spiral sampling pattern for smooth bokeh
        for (int i = 1; i < SAMPLES; i++) {
            float angle = float(i) * GOLDEN_ANGLE;
            float radius = sqrt(float(i)) / sqrt(float(SAMPLES));
            
            vec2 offset = vec2(cos(angle), sin(angle)) * radius * blur * 0.05;
            vec2 sampleUV = uv + offset;
            
            float sampleDepth = texture(uDepthTexture, sampleUV).r;
            float sampleBlur = abs(sampleDepth - uFocalDepth) / uFocalRange;
            
            // Weight based on depth similarity
            float weight = 1.0 / (1.0 + abs(depth - sampleDepth) * 10.0);
            
            color += texture(uTexture, sampleUV) * weight;
            totalWeight += weight;
        }
        
        return color / totalWeight;
    }
    
    void main() {
        FragColor = depthOfField(vTexCoord);
    }
    """
    
    # AI Enhancement Shader
    AI_ENHANCE_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform sampler2D uDetailTexture;
    uniform float uSharpness;
    uniform float uContrast;
    uniform float uSaturation;
    uniform float uDetail;
    uniform vec2 uResolution;
    
    // High-quality sharpening kernel
    vec4 sharpen(vec2 uv) {
        vec2 step = 1.0 / uResolution;
        
        vec4 center = texture(uTexture, uv);
        vec4 top = texture(uTexture, uv + vec2(0.0, -step.y));
        vec4 bottom = texture(uTexture, uv + vec2(0.0, step.y));
        vec4 left = texture(uTexture, uv + vec2(-step.x, 0.0));
        vec4 right = texture(uTexture, uv + vec2(step.x, 0.0));
        
        // Laplacian filter
        vec4 laplacian = 4.0 * center - top - bottom - left - right;
        
        return center + laplacian * uSharpness;
    }
    
    // Contrast and saturation adjustment
    vec3 adjustColor(vec3 color) {
        // Contrast
        color = (color - 0.5) * uContrast + 0.5;
        
        // Saturation
        float gray = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color = mix(vec3(gray), color, uSaturation);
        
        return color;
    }
    
    void main() {
        vec4 color = sharpen(vTexCoord);
        
        // Add detail from AI upscaling
        vec4 detail = texture(uDetailTexture, vTexCoord);
        color.rgb = mix(color.rgb, detail.rgb, uDetail);
        
        // Color adjustments
        color.rgb = adjustColor(color.rgb);
        
        // Preserve alpha
        FragColor = vec4(color.rgb, texture(uTexture, vTexCoord).a);
    }
    """
    
    # Particle System Shader
    PARTICLE_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    in vec2 vPosition;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform float uTime;
    uniform int uParticleCount;
    uniform vec2 uResolution;
    uniform vec3 uParticleColor;
    
    // Hash function for randomness
    float hash(vec2 p) {
        p = fract(p * vec2(234.34, 435.345));
        p += dot(p, p + 34.23);
        return fract(p.x * p.y);
    }
    
    // Particle generation
    vec4 particle(vec2 uv, float id) {
        float t = uTime * 0.5 + id * 6.283;
        
        // Particle position
        vec2 pos = vec2(
            sin(t * 1.3 + id * 2.4) * 0.4,
            cos(t * 0.7 + id * 1.7) * 0.4
        );
        
        // Add some noise
        pos += vec2(
            hash(vec2(id, t)) - 0.5,
            hash(vec2(t, id)) - 0.5
        ) * 0.2;
        
        // Distance to particle
        float dist = length(uv - pos - 0.5);
        
        // Particle size with pulsing
        float size = 0.002 + sin(t * 4.0 + id * 3.0) * 0.001;
        
        // Soft particle edge
        float particle = smoothstep(size, 0.0, dist);
        
        return vec4(uParticleColor * particle, particle);
    }
    
    void main() {
        vec4 texColor = texture(uTexture, vTexCoord);
        vec4 particles = vec4(0.0);
        
        // Render multiple particles
        for (int i = 0; i < uParticleCount; i++) {
            vec4 p = particle(vTexCoord, float(i) / float(uParticleCount));
            particles.rgb += p.rgb * p.a;
            particles.a = max(particles.a, p.a);
        }
        
        // Blend particles with texture
        FragColor = mix(texColor, vec4(particles.rgb, 1.0), particles.a * 0.7);
    }
    """
    
    # Distortion Effect
    DISTORTION_SHADER = """
    #version 330 core
    
    in vec2 vTexCoord;
    out vec4 FragColor;
    
    uniform sampler2D uTexture;
    uniform sampler2D uDistortionMap;
    uniform float uTime;
    uniform float uStrength;
    uniform vec2 uResolution;
    
    void main() {
        // Sample distortion map
        vec2 distortion = texture(uDistortionMap, vTexCoord + uTime * 0.01).rg - 0.5;
        
        // Apply distortion
        vec2 distortedUV = vTexCoord + distortion * uStrength;
        
        // Chromatic aberration on distortion
        float r = texture(uTexture, distortedUV + vec2(0.001, 0.0)).r;
        float g = texture(uTexture, distortedUV).g;
        float b = texture(uTexture, distortedUV - vec2(0.001, 0.0)).b;
        float a = texture(uTexture, distortedUV).a;
        
        FragColor = vec4(r, g, b, a);
    }
    """
    
    @staticmethod
    def get_effects() -> Dict[str, ShaderEffect]:
        """Get all available shader effects"""
        return {
            "neural_glow": ShaderEffect(
                name="Neural Glow",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.NEURAL_GLOW_SHADER,
                uniforms={
                    "uTime": 0.0,
                    "uIntensity": 1.0,
                    "uGlowColor": (0.0, 0.7, 1.0),
                    "uResolution": (1920, 1080)
                },
                description="Animated neural network glow effect"
            ),
            "chromatic_aberration": ShaderEffect(
                name="Chromatic Aberration",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.CHROMATIC_ABERRATION_SHADER,
                uniforms={
                    "uAmount": 5.0,
                    "uDirection": (1.0, 1.0),
                    "uResolution": (1920, 1080)
                },
                description="RGB channel separation effect"
            ),
            "depth_of_field": ShaderEffect(
                name="Depth of Field",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.DEPTH_OF_FIELD_SHADER,
                uniforms={
                    "uFocalDepth": 0.5,
                    "uFocalRange": 0.2,
                    "uBlurAmount": 1.0,
                    "uResolution": (1920, 1080)
                },
                description="Professional depth of field blur"
            ),
            "ai_enhance": ShaderEffect(
                name="AI Enhancement",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.AI_ENHANCE_SHADER,
                uniforms={
                    "uSharpness": 0.5,
                    "uContrast": 1.2,
                    "uSaturation": 1.1,
                    "uDetail": 0.3,
                    "uResolution": (1920, 1080)
                },
                description="AI-powered image enhancement"
            ),
            "particle_overlay": ShaderEffect(
                name="Particle Overlay",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.PARTICLE_SHADER,
                uniforms={
                    "uTime": 0.0,
                    "uParticleCount": 50,
                    "uParticleColor": (0.8, 0.9, 1.0),
                    "uResolution": (1920, 1080)
                },
                description="Dynamic particle system overlay"
            ),
            "distortion": ShaderEffect(
                name="Distortion",
                vertex_source=ShaderLibrary.BASE_VERTEX_SHADER,
                fragment_source=ShaderLibrary.DISTORTION_SHADER,
                uniforms={
                    "uTime": 0.0,
                    "uStrength": 0.1,
                    "uResolution": (1920, 1080)
                },
                description="Heat wave distortion effect"
            )
        }


class ShaderManager:
    """Manages shader compilation and caching"""
    
    def __init__(self):
        self.shader_cache: Dict[str, QOpenGLShaderProgram] = {}
        self.active_effects: Dict[str, bool] = {}
        
    def compile_shader(self, effect: ShaderEffect) -> Optional[QOpenGLShaderProgram]:
        """Compile and cache a shader effect"""
        if effect.name in self.shader_cache:
            return self.shader_cache[effect.name]
            
        program = QOpenGLShaderProgram()
        
        # Compile vertex shader
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex,
            effect.vertex_source
        ):
            print(f"Vertex shader compilation failed: {program.log()}")
            return None
            
        # Compile fragment shader
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment,
            effect.fragment_source
        ):
            print(f"Fragment shader compilation failed: {program.log()}")
            return None
            
        # Link program
        if not program.link():
            print(f"Shader linking failed: {program.log()}")
            return None
            
        self.shader_cache[effect.name] = program
        return program
        
    def set_uniforms(self, program: QOpenGLShaderProgram, uniforms: Dict[str, any]):
        """Set shader uniforms"""
        for name, value in uniforms.items():
            location = program.uniformLocation(name)
            if location == -1:
                continue
                
            if isinstance(value, (int, float)):
                program.setUniformValue(location, value)
            elif isinstance(value, tuple) and len(value) == 2:
                program.setUniformValue(location, value[0], value[1])
            elif isinstance(value, tuple) and len(value) == 3:
                program.setUniformValue(location, value[0], value[1], value[2])
            elif isinstance(value, np.ndarray):
                if value.shape == (4, 4):
                    program.setUniformValue(location, value)
                    
    def cleanup(self):
        """Clean up shader resources"""
        self.shader_cache.clear()