#include "ocean.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <slime/renderer/shader.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

using namespace slime;
using namespace std;

Ocean::Ocean()
{
    initX = 0.0f;
    initY = 0.0f;
    initZ = 0.0f;
    OceanSize = 32;
}

Ocean::Ocean(float _initX, float _initY, float _initZ, int _OceanSize)
{
    initX = _initX;
    initY = _initY;
    initZ = _initZ;
    OceanSize = _OceanSize;
}

Ocean::~Ocean()
{
}

void Ocean::setup()
{
    cout << "setup Ocean" << endl;
    this->shader = new Shader("./shaders/Ocean.vert", "./shaders/Ocean.frag");

    shader->use();

    /* Set Uniforms */
    shader->setInt("texture0", 0);
    shader->setVec3("material.ambient", 0.1f, 0.6f, 0.3f);
    shader->setVec3("material.diffuse", 0.1f, 0.6f, 0.3f);
    shader->setVec3("material.specular", 0.5f, 0.5f, 0.5f);
    shader->setFloat("material.shininess", 32.0f);

    /* Calculate Vertex Positions */
    float *vertices = new float[3 * OceanSize * OceanSize];

    for (int i = 0; i < OceanSize; i++)
    {
        for (int j = 0; j < OceanSize; j++)
        {
            float &vertX = vertices[i * 3 * OceanSize + j * 3];
            float &vertY = vertices[i * 3 * OceanSize + j * 3 + 1];
            float &vertZ = vertices[i * 3 * OceanSize + j * 3 + 2];
            vertX = i - OceanSize / 2;
            vertY = 0;
            vertZ = j - OceanSize / 2;
        }
    }

    /* Calculate Indices */

    unsigned int *indices = new unsigned int[(OceanSize - 1) * (OceanSize - 1) * 2 * 3];
    int current = 0;
    for (int i = 0; i < OceanSize - 1; i++)
    {
        for (int j = 0; j < OceanSize - 1; j++)
        {
            indices[current++] = i * OceanSize + j;
            indices[current++] = i * OceanSize + j + OceanSize;
            indices[current++] = i * OceanSize + j + OceanSize + 1;

            indices[current++] = i * OceanSize + j;
            indices[current++] = i * OceanSize + j + OceanSize + 1;
            indices[current++] = i * OceanSize + j + 1;
        }
    }

    /* VAO, VBO, EBO */
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * OceanSize * OceanSize, vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * (OceanSize - 1) * (OceanSize - 1) * 2 * 3, indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    // glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Ocean::render()
{
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // cout << "render Ocean" << endl;

    /* Transform */
    shader->use();

    glm::mat4 model = glm::mat4(1.0f);

    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));
    model = glm::translate(model, glm::vec3(initX, initY, initZ));

    glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(model)));

    glm::vec3 lightPos(1.1f, 3.0f, 2.0f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 objectColor(0.1f, 1.0f, 0.3f);
    shader->setVec3("lightPos", lightPos);
    shader->setVec3("viewPos", cameraPos);
    shader->setVec3("lightColor", lightColor);
    shader->setVec3("objectColor", objectColor);
    shader->setFloat("time", glfwGetTime());

    int modelLoc = glGetUniformLocation(shader->getID(), "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    int normalMatrixLoc = glGetUniformLocation(shader->getID(), "normalMatrix");
    glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));
    int viewLoc = glGetUniformLocation(shader->getID(), "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    int projectionLoc = glGetUniformLocation(shader->getID(), "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    /* Draw */
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, (OceanSize - 1) * (OceanSize - 1) * 2 * 3, GL_UNSIGNED_INT, 0);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Ocean::clear()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // glDeleteBuffers(1, &EBO);
}

void Ocean::updateView(glm::mat4 _view)
{
    this->view = _view;
}

void Ocean::updateProjection(glm::mat4 _projection)
{
    this->projection = _projection;
}

void Ocean::updateCameraPos(glm::vec3 _cameraPos)
{
    this->cameraPos = _cameraPos;
}