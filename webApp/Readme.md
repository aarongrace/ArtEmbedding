# Clash of Colonies

## Executive Summary

Ants are fascinating species with a complex social organization. A simulation game featuring ant colonies can be both entertaining and educational in terms of sociological behaviors and biology.

---

## Problem Overview

- How to effectively simulate a colony of ants?
- How to create a web-based solution that allows users to have their own colonies, join clans, trade resources, and send messages?
- How to store data securely and ensure consistency of game data across different play sessions?

---

## Proposed Solution

A web application using **React** for the frontend. User inputs will be sent to the backend, powered by **FastAPI**, and data will be stored securely in **MongoDB**.

---

## Scope of Work

This project seeks to simulate a colony of ants in terms of reproduction, resource gathering, and colony expansion.

---

# Descriptions and Implementations of Major Systems

## Core Game


## Users/Profile


## Clans

## Trades

## Admin Functions

## Navigation

## Shop






### React UI

#### Dashboard
- Displays several components:
  - **Resource Panel**
  - **Reproduction Panel**
  - **Map**
  - **Basic User and Clan Info**

#### Sign-In Page
- Users are prompted to sign in.
- If they do not have an account, they can create one through a separate sign-up page.

#### Clan View
- Players can add clans and initiate trades.
- Clan leaders have extra options.

#### Profile View
- Players can customize their name, details, and profile photo.
- Players can download a backup of their colony or import a backup.

#### About View
- A guide for players to understand how to play the game.

#### Units View
- Displays unit information (adults and broods) and allows for operations such as changing tasks or removing specific ants from the colony.

#### Store View
- Allows for cosmetic and gameplay upgrades.

#### Admin View
- The first admin is hardcoded.
- Admins can:
  - Make other players admins.
  - Ban users.
  - Change game variables.
  - Impersonate players.

#### Leaderboard
- Displays the colony size of each player.

---

### Backend Setup

The backend is built with **FastAPI**, which handles all routes and business logic.

#### FastAPI Router
- Routes requests from the frontend to the backend.
- Different routers are set up for:
  - Colony information
  - User information
  - Clan information

---

### MongoDB Database

The database consists of four collections:

1. **User Collection**:
   - Fields: `id`, `name`, `region`, `clan`, `images`, `createdDate`, `password`, `email`, `role`.

2. **Colony Collection**:
   - Fields: `id` (same as user ID), `name`, `ants`, `food`, `sand`, `age`, `map`, `perkPurchased`.

3. **Store Collection**:
   - Fields: `id`, `cosmetic or gameplay`, `associated bonus`.

4. **Clan Collection**:
   - Fields: `users` (separated into leader/officer/member), `clan score`.

---

### Tests

A series of manual tests and unit tests will be implemented for operations such as:
- Creation of users.
- Game functions such as the creation of new ants.
- Creation and deletion of clans.

---

### Timeline

| Week | Tasks                                                                 |
|------|----------------------------------------------------------------------|
| 1    | Backend setup, objects and interfaces, unit tests                   |
| 2    | Core game logic                                                     |
| 3    | Users, clans, profiles, and admin functions                         |
| 4    | Store, units view, leaderboard, debugging                           |

---


## Development Notes

The conversion between camel case and snake case is NOT automatic when sending data between frontend and backend. Please only use camel case for all class fields relating to backend data such as tasks or units!!! Mismatches in field names will result in **422 Unprocessable Entity** when sending objects.

### Coordinate System

The coordinates are set up so that (1,1) is in the lower right quadrant, and (-1,-1) is in the top left quadrant, and (0,0) is in the center. It is conceived this way so to allow expansion on all sides and to accommodate for the fact that y coordinates for display and for handling inputs start from the top. 

### npm Modules

If you want to install a npm module for the frontend, you need to cd to the frontend folder and install it there with npm install .... The npm package.json and node modules in the root folder are only used for concurrently and installing packages there might confuse VSCode into thinking that certain modules are enabled for the frontend when they are not.

## Todos

### Dashboard
- add a fourth picture for fruit spawns
- add background music and sound effects
- Create sprites for ant attack.
- Make task icons hoverable. When hovered, the icons should display a range indicator or a path indicator for ants doing that task. The hover can be implemented as part of the global settings store.
- Add icons for food/debris and ant production.
- add a custom cursor display

---

## Versions

### 2.0
- The whole project has been completely refactored such that all game behavior-related functions are relegated to the backend.
- Instead of creating new units and sending them to the backend, the new approach is to have the frontend send a message to advance the time cycle to the main router, which then deals with the logic.
- This is so that Unit objects would not be passed back and forth between the frontend and the backend, which causes endless headaches with Pydantic, as the frontend class declarations do not clearly line up with those from the backend. Deserialization and serialization work extremely poorly when dealing with abstract classes and inheritance, as Pydantic requires precision.

### 3.0
- Refactored the `Colony` model:
  - The `id` field now matches the `userId` and serves as the unique identifier for colonies.
  - All fields in the `Colony` model are now camel case.
  - Added an `initialize_default` method to create a default colony with 5 ants, 100 food, and 100 sand.
- Updated the backend to reflect the changes in the `Colony` model.
- Improved consistency between frontend and backend models by enforcing camel case across all fields.
- Added new features to the **Clan View**, **Admin View**, and **Store View**.

# Known Bugs
- Refreshing the page at an unlucky time might cause put requests to the colony router to malfunction and mess up the ant array. Only happened once so far