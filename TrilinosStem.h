#ifndef TRILINOS_STEM_H_
#define TRILINOS_STEM_H_

#include "Teuchos_RCPDecl.hpp"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"

#include "Tpetra_Map.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_DefaultPlatform.hpp"

#include "Ifpack2_Diagonal.hpp"

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
//#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_Parameters.hpp>
#include "BelosOperatorT.hpp"

class TrilinosStem {
    typedef double Scalar;
    typedef int Ordinal;
    typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
    typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType Node;
    //typedef Xpetra::EpetraNode Node;
    //typedef Xpetra::DefaultPlatform::DefaultPlatformType::NodeType Node
    typedef Tpetra::Map<Ordinal,Ordinal,Node> Map;
    typedef Xpetra::Map<Ordinal,Ordinal,Node> XMap;
    typedef Tpetra::CrsMatrix<Scalar,Ordinal,Ordinal,Node> Matrix;
    //typedef Xpetra::Matrix<Scalar,Ordinal,Ordinal,Node> Matrix;
    typedef Tpetra::Vector<Scalar,Ordinal,Ordinal,Node> Vector;
    typedef Tpetra::MultiVector<Scalar,Ordinal,Ordinal,Node> MultiVector;
    typedef Xpetra::MultiVector<Scalar,Ordinal,Ordinal,Node> XMultiVector;
    //typedef Tpetra::Operator<Scalar,Ordinal,Ordinal,Node> Operator;
    typedef Belos::OperatorT<MultiVector> Operator;

    public:
        TrilinosStem();
        static void initialise( int nx, int ny, int nz,
                int localNx, int localNy, int localNz,
                int local_xmin, int local_xmax,
                int local_ymin, int local_ymax,
		int local_zmin, int local_zmax,
		double dx,
		double dy,
		double dz,
                int tl_max_iters, double tl_eps);
        static void solve(int nx, int ny, int nz,
                int local_xmin, int local_xmax,
                int local_ymin, int local_ymax,
		int local_zmin, int local_zmax,
                int global_xmin, int global_xmax,
                int global_ymin, int global_ymax,
		int global_zmin, int global_zmax,
                int* numIters,
                double rx,
                double ry,
		double rz,
                double* Kx,
                double* Ky,
		double* Kz,
                double* u0);
        static void finalise();
    private:
        static Teuchos::RCP<const Map> map;
        static Teuchos::RCP<const XMap> xmap;
        static Teuchos::RCP<XMultiVector> xcoordinates;
        static Teuchos::RCP<Matrix> A;
        static Teuchos::RCP<Vector> b;
        static Teuchos::RCP<Vector> x;

        static Teuchos::RCP<Teuchos::ParameterList> solverParams;
        static Teuchos::RCP<Belos::LinearProblem<double, MultiVector, Operator> > problem;
        static Teuchos::RCP<Belos::SolverManager<double, MultiVector, Operator> > solver;
        //static Teuchos::RCP<Ifpack2::Preconditioner<Scalar, Ordinal, Ordinal, Node> > preconditioner;

        static int* myGlobalIndices_;
        static int numLocalElements_;
        static int MyPID;
};

#endif
