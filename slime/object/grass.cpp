#include "grass.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <slime/renderer/shader.h>
#include <stb_image.h>

using namespace slime;
using namespace std;

Grass::Grass()
{
}

Grass::Grass(float _initX, float _initY, float _initZ)
{
    initX = _initX;
    initY = _initY;
    initZ = _initZ;
}

Grass::~Grass()
{
}

void Grass::setup()
{
    cout << "setup grass" << endl;
    this->shader = new Shader("./shaders/grass.vert", "./shaders/grass.frag");

    /* VAO, VBO */

    float vertices[] = {
        -0.2f,
        0.0f,
        0.0f,
        0.2f,
        0.0f,
        0.0f,
        -0.18f,
        0.4f,
        0.0f,

        0.2f,
        0.0f,
        0.0f,
        -0.18f,
        0.4f,
        0.0f,
        0.18f,
        0.4f,
        0.0f,

        -0.18f,
        0.4f,
        0.0f,
        0.18f,
        0.4f,
        0.0f,
        -0.14f,
        0.7f,
        0.0f,

        0.18f,
        0.4f,
        0.0f,
        -0.14f,
        0.7f,
        0.0f,
        0.14f,
        0.7f,
        0.0f,

        -0.14f,
        0.7f,
        0.0f,
        0.14f,
        0.7f,
        0.0f,
        -0.08f,
        1.0f,
        0.0f,

        0.14f,
        0.7f,
        0.0f,
        -0.08f,
        1.0f,
        0.0f,
        0.08f,
        1.0f,
        0.0f,

        -0.08f,
        1.0f,
        0.0f,
        0.08f,
        1.0f,
        0.0f,
        0.0f,
        1.3f,
        0.0f,
    };

    // unsigned indices[] = {
    //     0, 1, 3,
    //     1, 2, 3};

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    // glEnableVertexAttribArray(1);
    // glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
    // glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Grass::render()
{
    // cout << "render cube" << endl;

    /* Transform */
    shader->use();

    glm::mat4 model = glm::mat4(1.0f);
    // glm::mat4 view = glm::mat4(1.0f);

    model = glm::scale(model, glm::vec3(0.3f, 0.5f, 0.3f));
    model = glm::translate(model, glm::vec3(initX, initY, initZ));
    // view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));

    int modelLoc = glGetUniformLocation(shader->getID(), "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    int viewLoc = glGetUniformLocation(shader->getID(), "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    int projectionLoc = glGetUniformLocation(shader->getID(), "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    /* Bind Textures and VAO */
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, texture0);
    // glActiveTexture(GL_TEXTURE1);
    // glBindTexture(GL_TEXTURE_2D, texture1);
    glBindVertexArray(VAO);

    /* Draw */
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Grass::clear()
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    // glDeleteBuffers(1, &EBO);
}

void Grass::updateView(glm::mat4 _view)
{
    this->view = _view;
}

void Grass::updateProjection(glm::mat4 _projection)
{
    this->projection = _projection;
}

void Grass::updateCameraPos(glm::vec3 _cameraPos)
{
    this->cameraPos = _cameraPos;
}