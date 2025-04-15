/**
 * Copyright (c) CTU  - All Rights Reserved
 * Created on: 4/30/20
 *     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
 */

#ifndef SIM_PHYSX_SCENE_H
#define SIM_PHYSX_SCENE_H

#include <Physics.h>
#include <BasePhysxPointer.h>
#include <RigidDynamic.h>
#include "RigidStatic.h"
#include "Aggregate.h"

class CollisionData {
public:
    physx::PxRigidActor* actor1;
    physx::PxRigidActor* actor2;
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    float impulse_magnitude;

    bool involves_actor(const RigidActor* actor) const {
        physx::PxRigidActor* actor_ptr = actor->get_physx_ptr();
        return actor_ptr == actor1 || actor_ptr == actor2;
    }
};

class Scene : public BasePhysxPointer<physx::PxScene>, public physx::PxSimulationEventCallback {
public:
    Scene(const physx::PxFrictionType::Enum &friction_type,
          const physx::PxBroadPhaseType::Enum &broad_phase_type,
          const std::vector<physx::PxSceneFlag::Enum> &scene_flags,
          size_t gpu_max_num_partitions,
          float gpu_dynamic_allocation_scale
    ) : BasePhysxPointer() {
        physx::PxSceneDesc sceneDesc(Physics::get().physics->getTolerancesScale());
        sceneDesc.cpuDispatcher = Physics::get().dispatcher;
        sceneDesc.cudaContextManager = Physics::get().cuda_context_manager;
        sceneDesc.filterShader = [](
            physx::PxFilterObjectAttributes attributes0, physx::PxFilterData filterData0,
            physx::PxFilterObjectAttributes attributes1, physx::PxFilterData filterData1,
            physx::PxPairFlags& pairFlags, const void* constantBlock, physx::PxU32 constantBlockSize) -> physx::PxFilterFlags
        {
            auto ret = physx::PxDefaultSimulationFilterShader(
                attributes0, filterData0,
                attributes1, filterData1,
                pairFlags, constantBlock, constantBlockSize
            );
            if (ret != physx::PxFilterFlag::eSUPPRESS) {
                pairFlags |= physx::PxPairFlag::eNOTIFY_TOUCH_FOUND;
                pairFlags |= physx::PxPairFlag::eNOTIFY_CONTACT_POINTS;
            }
            return ret;
        };
        sceneDesc.gravity = physx::PxVec3(0.0f, 9.81f, 0.0f);
        sceneDesc.simulationEventCallback = this;
        for (const auto &flag : scene_flags) {
            sceneDesc.flags |= flag;
        }
        sceneDesc.frictionType = friction_type;
        sceneDesc.broadPhaseType = broad_phase_type;
        sceneDesc.gpuMaxNumPartitions = gpu_max_num_partitions;
        sceneDesc.gpuDynamicsConfig.patchStreamSize *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.forceStreamCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.contactBufferCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.contactStreamSize *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.foundLostPairsCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.constraintBufferCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.heapCapacity *= gpu_dynamic_allocation_scale;
        sceneDesc.gpuDynamicsConfig.tempBufferCapacity *= gpu_dynamic_allocation_scale;

        set_physx_ptr(Physics::get().physics->createScene(sceneDesc));
    }

    /** @brief Simulate scene for given amount of time dt and fetch results with blocking. */
    void simulate(float dt) {
        collision_data.clear();
        get_physx_ptr()->simulate(dt);
        get_physx_ptr()->fetchResults(true);
        simulation_time += dt;
    }

    virtual void onContact(const physx::PxContactPairHeader& pairHeader,
                           const physx::PxContactPair* pairs,
                           physx::PxU32 nbPairs) override {
        for (physx::PxU32 i = 0; i < nbPairs; i++) {
            const physx::PxContactPair& cp = pairs[i];
            if (cp.events & physx::PxPairFlag::eNOTIFY_TOUCH_FOUND) {
                CollisionData data;
                data.actor1 = pairHeader.actors[0];
                data.actor2 = pairHeader.actors[1];
                physx::PxContactPairPoint point;
                if (cp.extractContacts(&point, 1) > 0) {
                    data.position = Eigen::Vector3f(point.position.x, point.position.y, point.position.z);
                    data.normal = Eigen::Vector3f(point.normal.x, point.normal.y, point.normal.z);
                    data.impulse_magnitude = point.impulse.magnitude();
                }
                collision_data.push_back(data);
            }
        }
    }
    virtual void onConstraintBreak(physx::PxConstraintInfo* constraints, physx::PxU32 count) override {}
    virtual void onWake(physx::PxActor** actors, physx::PxU32 count) override {}
    virtual void onSleep(physx::PxActor** actors, physx::PxU32 count) override {}
    virtual void onTrigger(physx::PxTriggerPair* pairs, physx::PxU32 count) override {}
    virtual void onAdvance(const physx::PxRigidBody*const* bodyBuffer, const physx::PxTransform* poseBuffer, const physx::PxU32 count) override {}

    const std::vector<CollisionData>& get_collision_data() const {
        return collision_data;
    }

    void add_actor(RigidActor actor) {
        get_physx_ptr()->addActor(*actor.get_physx_ptr());
    }

    void remove_actor(RigidActor actor) {
        get_physx_ptr()->removeActor(*actor.get_physx_ptr());
    }

    auto get_static_rigid_actors() {
        const auto n = get_physx_ptr()->getNbActors(physx::PxActorTypeFlag::eRIGID_STATIC);
        std::vector<physx::PxRigidActor *> actors(n);
        get_physx_ptr()->getActors(physx::PxActorTypeFlag::eRIGID_STATIC,
                                   reinterpret_cast<physx::PxActor **>(&actors[0]), n);
        return from_vector_of_physx_ptr<RigidActor>(actors);
    }

    auto get_dynamic_rigid_actors() {
        const auto n = get_physx_ptr()->getNbActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC);
        std::vector<physx::PxRigidDynamic *> actors(n);
        get_physx_ptr()->getActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC,
                                   reinterpret_cast<physx::PxActor **>(&actors[0]), n);
        return from_vector_of_physx_ptr<RigidDynamic, physx::PxRigidDynamic>(actors);
    }

    void add_aggregate(Aggregate agg) {
        get_physx_ptr()->addAggregate(*agg.get_physx_ptr());
    }

    auto get_aggregates() {
        const auto n = get_physx_ptr()->getNbAggregates();
        std::vector<physx::PxAggregate *> aggs(n);
        get_physx_ptr()->getAggregates(&aggs[0], n);
        return from_vector_of_physx_ptr<Aggregate>(aggs);
    }

public:
    double simulation_time = 0.;
    std::vector<CollisionData> collision_data;
};

#endif //SIM_PHYSX_SCENE_H
