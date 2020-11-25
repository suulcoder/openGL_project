import pygame
import numpy
import glm
import pyassimp
import site
import pyglet
from pyglet.gl import *
site.getsitepackages()


from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLUT import *
import sys
from PIL import Image

pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

vertex_shader = """
#version 430 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoords;

uniform mat4 theMatrix;
uniform vec3 light;

out float intensity;
out vec2 vertexTexcoords;

void main()
{
  vertexTexcoords = texcoords;
  intensity = dot(normal, normalize(light));
  gl_Position = theMatrix * vec4(position.x, position.y, position.z, 1.0);
}
"""

fragment_shader = """
#version 430 core
layout(location = 0) out vec4 fragColor;

in float intensity;
in vec2 vertexTexcoords;

uniform sampler2D tex;
uniform vec4 diffuse;
uniform vec4 ambient;

void main()
{

  fragColor = ambient + diffuse*texture(tex, vertexTexcoords) * intensity;
}
"""

shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

scene = pyassimp.load('./models/OBJ/challenger.obj')


texture_surface = pygame.image.load('./models/textures/grid.jpg')
texture_data = pygame.image.tostring(texture_surface, 'RGB')
width = texture_surface.get_width()
height = texture_surface.get_height()

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(
  GL_TEXTURE_2D,
  0,
  GL_RGB,
  width,
  height,
  0,
  GL_RGB,
  GL_UNSIGNED_BYTE,
  texture_data
)
glGenerateMipmap(GL_TEXTURE_2D)

data=pyglet.image.load('./models/textures/grid.jpg').get_data()
texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (width), (height), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

def loadskybox():
  glBindTexture(GL_TEXTURE_2D, 1)
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE,
  texture_data)

def drawskybox():
  glEnable(GL_TEXTURE_2D)
  glDisable(GL_DEPTH_TEST)
  glColor3f(1,1,1) # front face
  glBindTexture(GL_TEXTURE_2D, 1)
  glBegin(GL_QUADS)
  glTexCoord2f(0, 0)
  glVertex3f(-10.0, -10.0, -10.0)
  glTexCoord2f(1, 0)
  glVertex3f(10.0, -10.0, -10.0)
  glTexCoord2f(1, 1)
  glVertex3f(10.0, 10.0, -10.0)
  glTexCoord2f(0, 1)
  glVertex3f(-10.0, 10.0, -10.0)
  glEnd()
  glBindTexture(GL_TEXTURE_2D, 0)
  glEnable(GL_DEPTH_TEST)


def glize(node):
  # render
  for mesh in node.meshes:
    vertex_data = numpy.hstack([
      numpy.array(mesh.vertices, dtype=numpy.float32),
      numpy.array(mesh.normals, dtype=numpy.float32),
      numpy.array(mesh.texturecoords[0], dtype=numpy.float32),
    ])

    index_data = numpy.hstack(
      numpy.array(mesh.faces, dtype=numpy.int32),
    )

    vertex_buffer_object = glGenVertexArrays(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(6 * 4))
    glEnableVertexAttribArray(2)

    element_buffer_object = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)


    glUniform3f(
      glGetUniformLocation(shader, "light"),
      0, 15, 0
    )

    glUniform4f(
      glGetUniformLocation(shader, "diffuse"),
      0.3, 0.2, 0.2, 0.2
    )

    glUniform4f(
      glGetUniformLocation(shader, "ambient"),
      0, 0, 0, 0.5
    )


    glDrawElements(GL_TRIANGLES, len(index_data), GL_UNSIGNED_INT, None)

  for child in node.children:
    glize(child)
i = glm.mat4()

def createTheMatrix(x,y,counter):
  translate = glm.translate(i, glm.vec3(1, 0, 1))
  rotate = glm.rotate(i, glm.radians(counter*5), glm.vec3(0, 1, 0))
  scale = glm.scale(i, glm.vec3(1, 1, 1))

  model = translate * rotate * scale
  view = glm.lookAt(
    glm.vec3(0, 4, 15),
      glm.vec3(
        numpy.sin(glm.radians(x))*360, 
        numpy.sin(glm.radians(y))*360, 
        numpy.cos(glm.radians(x))*360 + numpy.sin(glm.radians(y))*360),
      glm.vec3(0, 1, 0)
    )
  projection = glm.perspective(glm.radians(20), 800/600, 0.1, 1000)

  return projection * view * model

glViewport(0, 0, 800, 600)

glEnable(GL_DEPTH_TEST)

running = True
rotating_camera_x = 0
rotating_camera_y = 0
cam_rot_x = 175
cam_rot_y = 200
counter = 0
while running:
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glClearColor(0.54, 0.61, 0.031, 1)

  glUseProgram(shader)

  theMatrix = createTheMatrix(cam_rot_x, cam_rot_y, counter)

  theMatrixLocation = glGetUniformLocation(shader, 'theMatrix')

  glUniformMatrix4fv(
    theMatrixLocation, # location
    1, # count
    GL_FALSE,
    glm.value_ptr(theMatrix)
  )

  # glDrawArrays(GL_TRIANGLES, 0, 3)
  # glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)

  glize(scene.rootnode)

  pygame.display.flip()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_e:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
      if event.key == pygame.K_f:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
      if event.key == pygame.K_a:
        rotating_camera_x = 1
      if event.key == pygame.K_d:
        rotating_camera_x = 2
      if event.key == pygame.K_w:
        rotating_camera_y = 1
      if event.key == pygame.K_s:
        rotating_camera_y = 2
    elif event.type == pygame.KEYUP:
      if event.key == pygame.K_a:
        rotating_camera_x = 0
      if event.key == pygame.K_d:
        rotating_camera_x = 0
      if event.key == pygame.K_w:
        rotating_camera_y = 0
      if event.key == pygame.K_s:
        rotating_camera_y = 0

    if(rotating_camera_x==1):
      cam_rot_x += 1
    elif(rotating_camera_x==2):
      cam_rot_x -= 1

    if(rotating_camera_y==1):
      cam_rot_y -= 1
    elif(rotating_camera_y==2):
      cam_rot_y += 1

  counter += 1
  if(counter==360):
    counter = 0
  clock.tick(0)